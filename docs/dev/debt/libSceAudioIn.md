# libSceAudioIn technical debt

### Silent stubs

- [sceAudioInInit](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - returns 0 without library-wide setup, as `sceAudioOutInit` does; each `sceAudioInOpen` opens its own SDL capture device (on macOS only with SDL's file-backed `disk` driver, see Functional)

### Unknown function info

- [sceAudioInIsPadEmbeddedMic](../../../core/libs/prx/libSceAudioIn/Export.cpp) - the PS5 module uses the low handle byte as a port index and returns whether its resolved device type equals 8, propagating negative device errors. Host capture devices have no corresponding controller-microphone classification; the protected throwing body remains unchanged.
- [sceAudioInDeviceIdOpen](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInDeviceIdAsyncOpen](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInSetUsbGain](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInIsSharedDevice](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInDeviceIdHqOpen](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInDeviceHqOpen](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInSetConnections](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInSetPortConnections](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInSetPortStatuses](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInGetHandleStatusInfo](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInOpenEx](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInHqOpenEx](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInDeviceOpenEx](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInChangeAppModuleState](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInSetSparkParam](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInSetSparkSideTone](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInSetCompressorPreGain](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInSetNoiseGateThreshold](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInSetMicLevel](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInSetDeviceMuteStatus](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInSetDevConnection](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInCountPorts](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInSetUserMute](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInVmicCreate](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInVmicDestroy](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInVmicWrite](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInVmicGetStatus](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInVmicSetMute](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInSetMode](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInSetMode2](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInExtCtrl](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature
- [sceAudioInInputs](../../../core/libs/prx/libSceAudioIn/Export.cpp) (libSceAudioIn) - unknown signature

### Functional

- [sceAudioInHqOpen, sceAudioInAsyncOpen](../../../core/libs/prx/libSceAudioIn/Export.cpp) - PS5 wrapper validation requires 48000 Hz followed by exactly 128 samples for both calls; HqOpen then requires format 2. The host uses the same SDL capture path for both, so native asynchronous scheduling and high-quality processing are not reproduced. Common port validation, device selection and host capture pacing remain incomplete.
- [sceAudioInGetGain, sceAudioInGetRerouteCount](../../../core/libs/prx/libSceAudioIn/Export.cpp) - GetGain writes 1.0f before handle validation and returns 0x80260105 for a null output, as confirmed by the PS5 module. Native handle bit checks and uninitialized/inactive port errors differ from the host port table. Native reroute counters follow device reassignment; the host opens each SDL device once and reports zero.
- On macOS no port opens the host's microphone ([Export.cpp](../../../core/libs/prx/libSceAudioIn/Export.cpp) `openDevice`): a Core Audio capture device would ask the user for microphone access and then record them although they never asked for it. Every port opens without a device, as on a console with no microphone attached: `sceAudioInInput` returns silence at the port's pace and `sceAudioInGetSilentState` reports `SILENT_STATE_DEVICE_NONE` (1, the value shadPS4's `ORBIS_AUDIO_IN_SILENT_STATE_DEVICE_NONE` also uses). Only SDL's file-backed `disk` driver still captures (the `guest_audio_in_capture` test). Real microphone input is a future, opt-in feature (a user setting that enables Core Audio capture); test `guest_audio_in_host_microphone_unused`
- `sceAudioInOpen` and `sceAudioInHqOpen` still throw for inputs nobody has verified: parameter format `0x10`, and HqOpen at 256 samples or 16 kHz. [KytyPS5](https://github.com/KytyPS5/KytyPS5/blob/c86c8b4d9333a766cc8c4495f89f893bf35e6199/src/libs/audio.cpp) treats `0x10` as float mono and answers HqOpen's other sizes and rates with `INVALID_SIZE`/`INVALID_FREQ`, one reimplementation's reading, not a console measurement
