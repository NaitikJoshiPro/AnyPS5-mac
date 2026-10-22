# UPSYNC4 (upstream 661f8b7171) test twins: upstream execution tests of Vulkan-only internals that the Metal
# device does not have. PendingTwins.mm skips each with its reason.
list(APPEND agcMetalTwins
        # Recorder::HoldNextSubmit holds a Vulkan timeline-semaphore submit (16683be010); Metal has none.
        "tests/execution/IndirectArgumentsPendingProducer.cpp=specifics/macos/tests/PendingTwins.mm"
)

# Recorder.cpp modes that arrived with UPSYNC4 and were owed at GATE5 (TechnicalDebt metal-upsync4-owed-tests):
# --read-written-staging-only, --storage-write-back-only and --writeback-blocks-only, registered with RecorderTwin.mm.
list(APPEND agcMetalTwins
        "tests/Recorder.cpp=specifics/macos/tests/StorageWriteBackTwin.mm"
)
