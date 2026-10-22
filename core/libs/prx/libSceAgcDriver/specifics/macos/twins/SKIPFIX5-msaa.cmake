# SKIPFIX5 (multisample skips after UPSYNC5). MultisampledPassOrder.cpp captures vkCmdPipelineBarrier through
# Context.deviceProc; the twin checks the Metal form of the same order (render encoders, each begun after a queue
# barrier, MetalCommandStream.mm BeginRender) on the same draw sequence through the Metal device. SampleOrder.cpp builds
# its references with Vulkan images and pipelines; its twin runs the same checks on Metal textures (its own fill and
# readback passes) and through the Metal device's draws.
list(APPEND agcMetalTwins
        "tests/execution/SampleOrder.cpp=specifics/macos/tests/SampleOrderTwin.mm"
        "tests/execution/MultisampledPassOrder.cpp=specifics/macos/tests/MultisampledPassOrderTwin.mm"
)
# The guest sample order's reorder passes on Metal (overlay MetalSampleOrder.hpp; Graphics/src/SampleOrder.cpp branches to
# them for a Metal context): per-sample render passes in place of upstream's storage-image compute. Built wherever the
# device is (SampleOrder.cpp and VulkanDevice.cpp are both in agcDriverSources).
list(APPEND agcMetalTwins
        "Execution/src/VulkanDevice.cpp=specifics/macos/src/MetalSampleOrder.mm"
)
