#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_METALOBJECTS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_METALOBJECTS_HPP

// macOS only (specifics/macos/overlay, work package D2): the Metal objects the plain-C++ headers of
// the driver twins hold opaquely. They are defined, and destroyed by ObjectsDeleter, in the
// Objective-C++ twins (specifics/macos/src), so these headers stay includable from C++.

namespace AgcDriver::Graphics::Metal {

class CommandStream;
struct RecorderObjects;
struct BatchObjects;
struct CommandBatchObjects;
struct PoolObjects;

struct ObjectsDeleter {
    // Recorder.mm
    void operator()(RecorderObjects* objects) const noexcept;
    void operator()(BatchObjects* objects) const noexcept;
    // Resources.mm
    void operator()(CommandBatchObjects* objects) const noexcept;
    // BufferPool.mm
    void operator()(PoolObjects* objects) const noexcept;
};

}

#endif
