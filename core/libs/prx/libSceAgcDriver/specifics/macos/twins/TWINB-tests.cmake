# TWIN-B (M2 test twins for D2, D4 and D6 tests that skipped as "Metal twin not written yet"): each port is its
# own source. The tests/Recorder.cpp modes register with RecorderTwin.mm (RecorderTwins.h) and share
# RecorderTwinHarness.h; PooledBufferRangeTwin.mm replaces tests/execution/PooledBufferRange.cpp. The
# MultisampleTarget twin gives multisampled color targets a Metal host image (agc_multisample_guest_access).
list(APPEND agcMetalTwins
        "tests/Recorder.cpp=specifics/macos/tests/StorageUploadTwin.mm"
        "tests/Recorder.cpp=specifics/macos/tests/DrawSnapshotCapTwin.mm"
        "tests/Recorder.cpp=specifics/macos/tests/DrawInputsInPlaceTwin.mm"
        "tests/Recorder.cpp=specifics/macos/tests/MultisampleGuestAccessTwin.mm"
        # Recorder.cpp modes upstream added by v0.1.2 (0f2683cb7; D29).
        "tests/Recorder.cpp=specifics/macos/tests/UpsyncRecorderModes.mm"
        "tests/Recorder.cpp=specifics/macos/tests/CopiedDirectOrderingTwin.mm"
        # Recorder.cpp modes upstream added by fdefd7660.
        "tests/Recorder.cpp=specifics/macos/tests/VertexTargetRangeTwin.mm"
        # Recorder.cpp mode upstream added by ea3a2152d (UPSYNC3): a storage write-back split across two imports.
        "tests/Recorder.cpp=specifics/macos/tests/SplitWriteBackTwin.mm"
        "tests/execution/PooledBufferRange.cpp=specifics/macos/tests/PooledBufferRangeTwin.mm"
        "Graphics/src/MultisampleTarget.cpp=specifics/macos/src/MultisampleTarget.mm"
)
