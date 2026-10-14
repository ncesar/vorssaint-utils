// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vorssaint

import Accelerate
import AVFoundation
import CoreAudio
import Darwin

/// A flag the audio thread raises and another thread reads later.
final class RecorderAudioFlag {
    private var bits: Int32 = 0

    var value: Bool { OSAtomicAdd32Barrier(0, &bits) != 0 }

    func raise() {
        OSAtomicCompareAndSwap32Barrier(0, 1, &bits)
    }
}

/// Whether linear float audio holds anything but silence.
enum RecorderAudioProbe {
    static func containsSound(_ sampleBuffer: CMSampleBuffer) -> Bool {
        guard let description = CMSampleBufferGetFormatDescription(sampleBuffer),
              let stream = CMAudioFormatDescriptionGetStreamBasicDescription(description)?.pointee,
              stream.mFormatID == kAudioFormatLinearPCM,
              stream.mFormatFlags & kAudioFormatFlagIsFloat != 0,
              stream.mBitsPerChannel == 32
        else { return false }
        let heard = try? sampleBuffer.withAudioBufferList(blockBufferMemoryAllocator: nil,
                                                          flags: []) { list, _ in
            list.contains { containsSound($0) }
        }
        return heard ?? false
    }

    /// One vectorised pass and no allocation, so the audio thread can ask.
    static func containsSound(_ buffer: AudioBuffer) -> Bool {
        guard let samples = buffer.mData?.assumingMemoryBound(to: Float.self) else { return false }
        let count = Int(buffer.mDataByteSize) / MemoryLayout<Float>.size
        guard count > 0 else { return false }
        var peak: Float = 0
        vDSP_maxmgv(samples, 1, &peak, vDSP_Length(count))
        return peak > 0
    }
}

/// The sound of the Mac for a recording, read from the audio system as one
/// mix of every process but this one. A recording of one window hears only
/// the app that owns it, so the sound follows the picture.
///
/// ScreenCaptureKit hands over the same mix, but an app the volume mixer
/// adjusts reaches it twice: its own muted stream and this app's re-render
/// of it, about 30 ms apart, and excluding this app from the stream's filter
/// does not drop that re-render (measured 2026-09-01, issue #692). A process
/// tap that excludes this process hears each app once.
///
/// Reading a tap needs an audio permission of its own. It is asked for at
/// the first start and, while missing, answered with silence rather than an
/// error, so the session keeps the stream's sound as the written source until
/// this tap has heard sound over a whole recording.
///
/// Every mutable field is touched only on `queue`, except `heard`, which is
/// atomic, and `onSample`, which the caller sets once before `start`. That
/// confinement is what makes the reference safe to hand across threads.
final class RecorderSystemAudioTap: @unchecked Sendable {
    /// Called on the audio system's own thread, with the sound retimed onto
    /// the recording's clock. Set before `start`.
    var onSample: ((CMSampleBuffer) -> Void)?

    /// Called on this tap's own queue when a device or process change left it
    /// without a reader, so the recording can go back to the stream's sound
    /// instead of falling silent. Set before `start`.
    var onReaderLost: (() -> Void)?

    /// Whether any sample was more than silence. Read after `stop`.
    var heardSound: Bool { heard.value }

    private let queue: DispatchQueue
    private let heard = RecorderAudioFlag()
    /// The one app to hear, or nothing for the whole Mac.
    private let pid: pid_t?
    private var tapID = AudioObjectID(0)
    private var tapUID = ""
    private var tapChannels = 2
    /// The processes an app's tap was built from, to notice when they change.
    private var tapped: [AudioObjectID] = []
    private var clock: CMClock?
    private var aggregateID = AudioObjectID(0)
    private var ioProc: AudioDeviceIOProcID?
    private var hostDeviceUID: String?
    private var sampleRate: Double = 0
    private var deviceListener: AudioObjectPropertyListenerBlock?
    private var rateListener: AudioObjectPropertyListenerBlock?
    private var processListener: AudioObjectPropertyListenerBlock?
    private var stopped = false

    /// Destroying a device or a tap can park inside a broken audio path, so
    /// it never happens on a queue anything waits for. Serial, so the device
    /// that hosts a tap is always gone before the tap itself.
    private static let teardownQueue = DispatchQueue(label: "com.vorssaint.recorder.systemaudio.teardown",
                                                     qos: .utility)

    /// The tap, or nothing when the audio system refuses one. A `pid` narrows
    /// it to that app's sound.
    static func make(hearing pid: pid_t?) async -> RecorderSystemAudioTap? {
        guard #available(macOS 14.4, *) else { return nil }
        let queue = DispatchQueue(label: "com.vorssaint.recorder.systemaudio",
                                  qos: .userInitiated)
        return await withCheckedContinuation { continuation in
            queue.async {
                continuation.resume(returning: RecorderSystemAudioTap(queue: queue, pid: pid))
            }
        }
    }

    @available(macOS 14.4, *)
    private init?(queue: DispatchQueue, pid: pid_t?) {
        self.queue = queue
        self.pid = pid
        guard buildTap() else { return nil }
    }

    /// One app's processes for a recording of its window, or every process
    /// but this one.
    @available(macOS 14.4, *)
    private func buildTap() -> Bool {
        let description: CATapDescription
        if let pid {
            let objects = Self.processObjects(for: pid)
            guard !objects.isEmpty else { return false }
            tapped = objects
            description = CATapDescription(stereoMixdownOfProcesses: objects)
        } else {
            guard let ownProcess = Self.processObject(for: ProcessInfo.processInfo.processIdentifier)
            else { return false }
            description = CATapDescription(stereoGlobalTapButExcludeProcesses: [ownProcess])
        }
        description.name = "Vorssaint Recorder"
        description.isPrivate = true
        var tapID = AudioObjectID(0)
        guard AudioHardwareCreateProcessTap(description, &tapID) == noErr, tapID != 0 else {
            return false
        }
        self.tapID = tapID
        tapUID = description.uuid.uuidString
        var format = AudioStreamBasicDescription()
        tapChannels = Self.read(tapID, kAudioTapPropertyFormat, &format) && format.mChannelsPerFrame > 0
            ? Int(format.mChannelsPerFrame)
            : 2
        return true
    }

    deinit {
        let tapID = self.tapID
        let aggregateID = self.aggregateID
        let ioProc = self.ioProc
        Self.destroy(aggregateID: aggregateID, ioProc: ioProc, tapID: tapID)
    }

    // MARK: - Lifecycle

    /// Starts delivering and answers whether a reader is now running. The
    /// first start on a Mac may raise the system's audio permission prompt;
    /// until it is granted, the reader runs but every buffer is silent.
    /// A false answer means no reader exists, so its sound must come from
    /// elsewhere for this recording.
    @discardableResult
    func start(synchronizingTo clock: CMClock) async -> Bool {
        await withCheckedContinuation { continuation in
            queue.async { [self] in
                self.clock = clock
                if !stopped {
                    buildPipeline()
                    watchDefaultOutputDevice()
                    watchProcessList()
                }
                continuation.resume(returning: aggregateID != 0)
            }
        }
    }

    func stop() async {
        await withCheckedContinuation { continuation in
            queue.async { [self] in
                stopped = true
                if let deviceListener {
                    var address = Self.address(kAudioHardwarePropertyDefaultOutputDevice)
                    AudioObjectRemovePropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject),
                                                           &address, queue, deviceListener)
                    self.deviceListener = nil
                }
                if let processListener {
                    var address = Self.address(kAudioHardwarePropertyProcessObjectList)
                    AudioObjectRemovePropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject),
                                                           &address, queue, processListener)
                    self.processListener = nil
                }
                teardownPipeline()
                Self.destroy(aggregateID: 0, ioProc: nil, tapID: tapID)
                continuation.resume()
            }
        }
    }

    // MARK: - Pipeline

    /// The tap needs an aggregate device to be read through, and that device
    /// needs a real one beneath it for its clock: the default output, as the
    /// volume mixer does.
    private func buildPipeline() {
        guard aggregateID == 0, let clock,
              let hostUID = Self.hostDeviceUID() else { return }
        let aggregate: [String: Any] = [
            kAudioAggregateDeviceNameKey: "Vorssaint Recorder",
            kAudioAggregateDeviceUIDKey: UUID().uuidString,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceMainSubDeviceKey: hostUID,
            kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: hostUID]],
            kAudioAggregateDeviceTapListKey: [[
                kAudioSubTapUIDKey: tapUID,
                kAudioSubTapDriftCompensationKey: true,
            ]],
            kAudioAggregateDeviceTapAutoStartKey: true,
        ]
        var aggregateID = AudioObjectID(0)
        guard AudioHardwareCreateAggregateDevice(aggregate as CFDictionary, &aggregateID) == noErr,
              aggregateID != 0 else { return }
        let sampleRate = Self.nominalSampleRate(of: aggregateID)
        guard let format = Self.format(sampleRate: sampleRate, channels: tapChannels) else {
            Self.destroy(aggregateID: aggregateID, ioProc: nil, tapID: 0)
            return
        }

        // The audio thread touches only these captured values, never the
        // engine object, matching the mixer's realtime discipline.
        let channels = tapChannels
        let heard = self.heard
        let onSample = self.onSample
        let timescale = CMTimeScale(sampleRate.rounded())
        let hostClock = CMClockGetHostTimeClock()
        var ioProc: AudioDeviceIOProcID?
        let created = AudioDeviceCreateIOProcIDWithBlock(&ioProc, aggregateID, nil) {
            _, input, inputTime, output, _ in
            // The device beneath the aggregate would otherwise play whatever
            // this memory last held.
            MixerRender.silence(UnsafeMutableAudioBufferListPointer(output))
            let inputBuffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: input))
            guard let index = MixerRender.tapBufferIndex(in: inputBuffers, tapChannels: channels)
            else { return }
            let buffer = inputBuffers[index]
            if !heard.value, RecorderAudioProbe.containsSound(buffer) { heard.raise() }
            guard let onSample else { return }
            let hostTime = inputTime.pointee.mFlags.contains(.hostTimeValid)
                ? CMClockMakeHostTimeFromSystemUnits(inputTime.pointee.mHostTime)
                : CMClockGetTime(hostClock)
            let time = CMSyncConvertTime(hostTime, from: hostClock, to: clock)
            guard time.isValid,
                  let sample = Self.sampleBuffer(from: buffer, format: format,
                                                 timescale: timescale, time: time)
            else { return }
            onSample(sample)
        }
        guard created == noErr, let ioProc,
              AudioDeviceStart(aggregateID, ioProc) == noErr else {
            Self.destroy(aggregateID: aggregateID, ioProc: ioProc, tapID: 0)
            return
        }
        self.aggregateID = aggregateID
        self.ioProc = ioProc
        hostDeviceUID = hostUID
        self.sampleRate = sampleRate
        watchSampleRate(of: aggregateID)
    }

    private func teardownPipeline() {
        let aggregateID = self.aggregateID
        let ioProc = self.ioProc
        let rateListener = self.rateListener
        self.aggregateID = 0
        self.ioProc = nil
        self.rateListener = nil
        hostDeviceUID = nil
        guard aggregateID != 0 else { return }
        if let ioProc {
            AudioDeviceStop(aggregateID, ioProc)
        }
        if let rateListener {
            var address = Self.address(kAudioDevicePropertyNominalSampleRate)
            AudioObjectRemovePropertyListenerBlock(aggregateID, &address, queue, rateListener)
        }
        Self.destroy(aggregateID: aggregateID, ioProc: ioProc, tapID: 0)
    }

    /// Headphones that come and go, or a device that changes its rate for a
    /// call, would otherwise leave the rest of the recording silent or at
    /// the wrong pitch. Only a real change rebuilds, so the notifications a
    /// rebuild itself raises settle instead of looping.
    private func rebuildPipelineIfChanged() {
        guard !stopped, aggregateID != 0 else { return }
        let hostChanged = Self.hostDeviceUID() != hostDeviceUID
        let rateChanged = Self.nominalSampleRate(of: aggregateID) != sampleRate
        guard hostChanged || rateChanged else { return }
        teardownPipeline()
        buildPipeline()
        // Nothing came back, and with no aggregate left this tap can no longer
        // see a later change, so the rest of the recording needs the stream.
        if aggregateID == 0 { onReaderLost?() }
    }

    /// A browser makes its sound in a helper process that comes and goes with
    /// its tabs, and a tap hears only the processes it was built from, so an
    /// app's tap is made again around its processes as they are now. Only a
    /// real change rebuilds, because any app launching stirs the same list.
    private func rebuildTapIfProcessesChanged() {
        guard !stopped, aggregateID != 0, let pid, #available(macOS 14.4, *) else { return }
        let current = Self.processObjects(for: pid)
        guard !current.isEmpty, Set(current) != Set(tapped) else { return }
        teardownPipeline()
        Self.destroy(aggregateID: 0, ioProc: nil, tapID: tapID)
        tapID = 0
        if buildTap() { buildPipeline() }
        if aggregateID == 0 { onReaderLost?() }
    }

    private func watchProcessList() {
        guard pid != nil, processListener == nil else { return }
        var address = Self.address(kAudioHardwarePropertyProcessObjectList)
        let listener: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            self?.rebuildTapIfProcessesChanged()
        }
        if AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject),
                                               &address, queue, listener) == noErr {
            processListener = listener
        }
        // The app may have opened a helper since the tap was made.
        rebuildTapIfProcessesChanged()
    }

    private func watchDefaultOutputDevice() {
        guard deviceListener == nil else { return }
        var address = Self.address(kAudioHardwarePropertyDefaultOutputDevice)
        let listener: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            self?.rebuildPipelineIfChanged()
        }
        if AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject),
                                               &address, queue, listener) == noErr {
            deviceListener = listener
        }
    }

    private func watchSampleRate(of aggregateID: AudioObjectID) {
        var address = Self.address(kAudioDevicePropertyNominalSampleRate)
        let listener: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            self?.rebuildPipelineIfChanged()
        }
        if AudioObjectAddPropertyListenerBlock(aggregateID, &address, queue, listener) == noErr {
            rateListener = listener
        }
    }

    private static func destroy(aggregateID: AudioObjectID,
                                ioProc: AudioDeviceIOProcID?,
                                tapID: AudioObjectID) {
        guard aggregateID != 0 || tapID != 0 else { return }
        teardownQueue.async {
            if aggregateID != 0 {
                if let ioProc {
                    AudioDeviceDestroyIOProcID(aggregateID, ioProc)
                }
                AudioHardwareDestroyAggregateDevice(aggregateID)
            }
            if tapID != 0, #available(macOS 14.4, *) {
                AudioHardwareDestroyProcessTap(tapID)
            }
        }
    }

    // MARK: - Samples

    private static func format(sampleRate: Double, channels: Int) -> CMAudioFormatDescription? {
        guard sampleRate > 0, channels > 0 else { return nil }
        let bytesPerFrame = UInt32(channels * MemoryLayout<Float>.size)
        var stream = AudioStreamBasicDescription(mSampleRate: sampleRate,
                                                 mFormatID: kAudioFormatLinearPCM,
                                                 mFormatFlags: kAudioFormatFlagIsFloat
                                                     | kAudioFormatFlagIsPacked,
                                                 mBytesPerPacket: bytesPerFrame,
                                                 mFramesPerPacket: 1,
                                                 mBytesPerFrame: bytesPerFrame,
                                                 mChannelsPerFrame: UInt32(channels),
                                                 mBitsPerChannel: 32,
                                                 mReserved: 0)
        var description: CMAudioFormatDescription?
        guard CMAudioFormatDescriptionCreate(allocator: kCFAllocatorDefault,
                                             asbd: &stream,
                                             layoutSize: 0,
                                             layout: nil,
                                             magicCookieSize: 0,
                                             magicCookie: nil,
                                             extensions: nil,
                                             formatDescriptionOut: &description) == noErr
        else { return nil }
        return description
    }

    /// The samples copied out of the audio system's buffer, which is reused
    /// as soon as this cycle returns.
    private static func sampleBuffer(from buffer: AudioBuffer,
                                     format: CMAudioFormatDescription,
                                     timescale: CMTimeScale,
                                     time: CMTime) -> CMSampleBuffer? {
        let frames = MixerRender.frames(bytes: buffer.mDataByteSize, channels: buffer.mNumberChannels)
        guard frames > 0 else { return nil }
        var timing = CMSampleTimingInfo(duration: CMTime(value: 1, timescale: timescale),
                                        presentationTimeStamp: time,
                                        decodeTimeStamp: .invalid)
        var sampleBuffer: CMSampleBuffer?
        guard CMSampleBufferCreate(allocator: kCFAllocatorDefault,
                                   dataBuffer: nil,
                                   dataReady: false,
                                   makeDataReadyCallback: nil,
                                   refcon: nil,
                                   formatDescription: format,
                                   sampleCount: frames,
                                   sampleTimingEntryCount: 1,
                                   sampleTimingArray: &timing,
                                   sampleSizeEntryCount: 0,
                                   sampleSizeArray: nil,
                                   sampleBufferOut: &sampleBuffer) == noErr,
              let sampleBuffer
        else { return nil }
        var list = AudioBufferList(mNumberBuffers: 1, mBuffers: buffer)
        guard CMSampleBufferSetDataBufferFromAudioBufferList(
            sampleBuffer,
            blockBufferAllocator: kCFAllocatorDefault,
            blockBufferMemoryAllocator: kCFAllocatorDefault,
            flags: 0,
            bufferList: &list) == noErr
        else { return nil }
        return sampleBuffer
    }

    // MARK: - Audio system

    private static func address(_ selector: AudioObjectPropertySelector,
                                scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal)
        -> AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(mSelector: selector,
                                   mScope: scope,
                                   mElement: kAudioObjectPropertyElementMain)
    }

    private static func read<T>(_ object: AudioObjectID,
                                _ selector: AudioObjectPropertySelector,
                                _ value: inout T,
                                scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> Bool {
        var address = address(selector, scope: scope)
        var size = UInt32(MemoryLayout<T>.size)
        return withUnsafeMutablePointer(to: &value) { pointer in
            AudioObjectGetPropertyData(object, &address, 0, nil, &size,
                                       UnsafeMutableRawPointer(pointer)) == noErr
        }
    }

    /// Every audio process the app answers for. A browser makes its sound in
    /// a helper process, so a tap on the app's own pid alone would never hear
    /// it; the mixer gathers its rows the same way.
    @available(macOS 14.4, *)
    private static func processObjects(for pid: pid_t) -> [AudioObjectID] {
        var objects: [AudioObjectID] = []
        for object in AppVolumeMixer.audioProcessObjects() {
            var owner: pid_t = -1
            guard read(object, kAudioProcessPropertyPID, &owner), owner > 0,
                  owner == pid || ResponsibleProcess.regularAppOwner(of: owner)?.processIdentifier == pid
            else { continue }
            objects.append(object)
        }
        // An app that holds no audio connection yet still has a process
        // object of its own, which starts sounding when it does.
        if objects.isEmpty, let direct = processObject(for: pid) { objects.append(direct) }
        return objects
    }

    private static func processObject(for pid: pid_t) -> AudioObjectID? {
        var pid = pid
        var object = AudioObjectID(0)
        var address = address(kAudioHardwarePropertyTranslatePIDToProcessObject)
        var size = UInt32(MemoryLayout<AudioObjectID>.size)
        let status = withUnsafePointer(to: &pid) { pidPointer in
            AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address,
                                       UInt32(MemoryLayout<pid_t>.size), pidPointer,
                                       &size, &object)
        }
        guard status == noErr, object != 0 else { return nil }
        return object
    }

    private static func nominalSampleRate(of deviceID: AudioObjectID) -> Double {
        var sampleRate: Float64 = 0
        guard read(deviceID, kAudioDevicePropertyNominalSampleRate, &sampleRate), sampleRate > 0
        else { return 48_000 }
        return sampleRate
    }

    /// The device that clocks the aggregate: the current default output, the
    /// same choice the volume mixer makes. A device that also records puts
    /// its input in front of the tap in the buffer list, which the shared
    /// `MixerRender.tapBufferIndex` already steps over.
    private static func hostDeviceUID() -> String? {
        var defaultDevice = AudioObjectID(0)
        guard read(AudioObjectID(kAudioObjectSystemObject),
                   kAudioHardwarePropertyDefaultOutputDevice, &defaultDevice),
              defaultDevice != 0 else { return nil }
        var uid: CFString = "" as CFString
        guard read(defaultDevice, kAudioDevicePropertyDeviceUID, &uid) else { return nil }
        return uid as String
    }
}
