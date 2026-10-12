# Work package D5 (descriptors, samplers, pipelines, caches): upstream Vulkan source = Metal twin.
# PipelineCache.mm is the device's shader library and pipeline cache (MTL4Compiler + MTL4Archive,
# background compilation; src/MetalPipelineCache.h). D5RecorderModes.mm ports tests/Recorder.cpp's
# --pipeline-cache-only mode (agc_pipeline_cache) through RecorderTwins.h.
list(APPEND agcMetalTwins
        "Graphics/src/PipelineCache.cpp=specifics/macos/src/PipelineCache.mm"
        "tests/Recorder.cpp=specifics/macos/tests/D5RecorderModes.mm"
)
