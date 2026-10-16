# UPSYNC3 (upstream 721db55dc) test twins: Metal ports of the tests/Recorder.cpp modes upstream added since 0214069e3,
# and the upstream execution tests that build their own Vulkan instance. The image table tests (tests/execution/
# ImageTable*.cpp) run unchanged on Metal (MslAbi Version 15 image and sampler tables).
list(APPEND agcMetalTwins
        # --read-list-only, --space-pending-writes-only, --pending-write-snapshot-only, --queued-write-groups-only,
        # --pending-key-stores-only, --shadowed-key-stores-only, --kept-unit-settle-only.
        "tests/Recorder.cpp=specifics/macos/tests/Upsync3RecorderModesA.mm"
        # --sparse-run-import-only, --sparse-run-copied-only, --draw-snapshot-epoch-only, --slice-dirty-only,
        # --metadata-pass-only, --resident-present-only (tests/ResidentPresent.cpp), --tiled-resolve-only,
        # --pipeline-cache-file-only (over the Metal pipeline archive, as D5RecorderModes.mm maps --pipeline-cache-only).
        "tests/Recorder.cpp=specifics/macos/tests/Upsync3RecorderModesB.mm"
        # Second round (upstream 5d9dcc42d2): --import-watch-only (the decision checks; the probe refuses on Metal,
        # D19) and --sample-dumps-only (without the Vulkan CoverPass half).
        "tests/Recorder.cpp=specifics/macos/tests/Upsync3RecorderModesC.mm"
        # Builds its own Vulkan instance upstream; the twin runs it unchanged over the Metal image context.
        "tests/execution/PartiallyResidentSnapshot.cpp=specifics/macos/tests/PartiallyResidentSnapshotTwin.mm"
        # Second round: builds its own Vulkan instance upstream; Run runs unchanged over the Metal image context.
        "tests/execution/DepthSurfaceLastWriter.cpp=specifics/macos/tests/DepthSurfaceLastWriterTwin.mm"
        # Second round: native multisampled depth sampling has no Metal path yet; every mode skips naming it.
        "tests/execution/NativeMultisampleDepth.cpp=specifics/macos/tests/NativeMultisampleDepthTwin.mm"
        # The GPU presentation pass is Vulkan-only (metal-present-pass): the twin runs the letterbox checks and skips.
        "tests/PresentationPass.cpp=specifics/macos/tests/PresentationPassTwin.mm"
)
