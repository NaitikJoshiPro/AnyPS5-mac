#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALDEPTHSURFACES_H
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALDEPTHSURFACES_H

// Work package D6: the Metal side of the depth surface registry (DepthSurface.mm, the twin of
// Graphics/src/DepthSurface.cpp). Objective-C++ only.

#import <Metal/Metal.h>

#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include <vector>

namespace AgcDriver::Graphics::Metal {

// A depth/stencil surface for one render pass: its texture and the clears the pass's load actions
// must perform (a new surface, or an HTILE fast clear noted since its last use).
struct DepthSurfaceUse {
    id<MTLTexture> texture;
    bool clearDepth;
    bool clearStencil;
    float depth;
    std::uint8_t stencil;
};

// The surface of `target` on `device` in `format` (made, cleared, on first use; `created` tells the
// caller to make it resident). Takes the pending clears. `depthWritten` (upstream 84e3b73f16): the pass
// writes depth, so a single-sample 32-bit float Z-swizzled plane's guest memory goes stale until
// FlushDepthPlanes stores it (read-only binds pass false).
DepthSurfaceUse AcquireDepthSurface(id<MTLDevice> device, const DepthTarget& target, MTLPixelFormat format, bool* created, bool depthWritten = true);
// The textures of the device's surfaces (ClearDepthSurfaces drops them at teardown).
std::vector<id<MTLTexture>> DepthSurfaceTextures(id<MTLDevice> device);

}

#endif
