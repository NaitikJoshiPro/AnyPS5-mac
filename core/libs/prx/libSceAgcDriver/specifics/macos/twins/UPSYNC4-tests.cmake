# UPSYNC4 (upstream 661f8b7171) test twins: upstream execution tests of Vulkan-only internals that the Metal
# device does not have. PendingTwins.mm skips each with its reason.
list(APPEND agcMetalTwins
        # Recorder::HoldNextSubmit holds a Vulkan timeline-semaphore submit (16683be010); Metal has none.
        "tests/execution/IndirectArgumentsPendingProducer.cpp=specifics/macos/tests/PendingTwins.mm"
)
