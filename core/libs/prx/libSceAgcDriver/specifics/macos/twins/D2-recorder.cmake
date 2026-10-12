# Work package D2 (recorder and buffer primitives): upstream Vulkan source = Metal twin.
# MetalCommandStream.mm (the command stream and helper kernels every batch records through) rides
# with Resources.cpp, which every target compiling a D2 original also compiles. The two tests are
# Vulkan-mock tests of the recorder and the buffer pool, replaced by Metal twins of the same checks.
list(APPEND agcMetalTwins
        "Graphics/src/Recorder.cpp=specifics/macos/src/Recorder.mm"
        "Graphics/src/Resources.cpp=specifics/macos/src/Resources.mm"
        "Graphics/src/Resources.cpp=specifics/macos/src/MetalCommandStream.mm"
        "Graphics/src/BufferPool.cpp=specifics/macos/src/BufferPool.mm"
        "Graphics/src/BufferAddress.cpp=specifics/macos/src/BufferAddress.mm"
        "tests/RecorderKeptBytes.cpp=specifics/macos/tests/RecorderKeptBytesMetal.mm"
        "tests/BufferPool.cpp=specifics/macos/tests/BufferPoolMetal.mm"
)
