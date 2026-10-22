# SKIPFIX5 (multisample skips after UPSYNC5). MultisampledPassOrder.cpp captures vkCmdPipelineBarrier through
# Context.deviceProc; the twin checks the Metal form of the same order (render encoders, each begun after a queue
# barrier, MetalCommandStream.mm BeginRender) on the same draw sequence through the Metal device. SampleOrder.cpp builds
# its references with Vulkan images and pipelines; its twin checks the capability answers its modes gate on and skips
# naming the Metal sample-order gaps (metal-sample-order, metal-sample-location-grid; not allowlisted).
list(APPEND agcMetalTwins
        "tests/execution/SampleOrder.cpp=specifics/macos/tests/SampleOrderTwin.mm"
        "tests/execution/MultisampledPassOrder.cpp=specifics/macos/tests/MultisampledPassOrderTwin.mm"
)
