# Work package D3 (guest memory, imports, BDA): host imports are no-copy Metal buffers.
list(APPEND agcMetalTwins
        "Graphics/src/GuestBufferMemory.cpp=specifics/macos/src/GuestBufferMemory.mm"
)
