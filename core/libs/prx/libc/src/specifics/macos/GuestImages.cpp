#ifdef __APPLE__

#include "prx/libc/include/specifics/macos/GuestImages.hpp"
#include <mach-o/dyld.h>
#include <mach-o/getsect.h>
#include <mach-o/loader.h>
#include <atomic>
#include <cstring>
#include <mutex>
#include <vector>

namespace {

constexpr std::uint64_t MetaVersion = 2;
constexpr std::size_t LoadBiasOffset = 0x08;
constexpr std::size_t HeaderCountOffset = 0x58;
constexpr std::size_t HeadersOffset = 0x60;

std::atomic<unsigned long long> imagesAdded {0};
std::atomic<unsigned long long> imagesRemoved {0};
// Process-owned: guest code still walks the images from atexit handlers and late static destructors
// (guest_memory_lifetime), after a namespace-scope mutex would have been destroyed (macOS then fails
// the lock with EINVAL), so the registry is never destroyed.
struct RegisteredImages {
    std::mutex mutex;
    std::vector<dl_phdr_info> images;
};

RegisteredImages& Registered() {
    static auto* registered = new RegisteredImages;
    return *registered;
}

void OnImageAdded(const mach_header*, intptr_t) {
    imagesAdded.fetch_add(1, std::memory_order_relaxed);
}

void OnImageRemoved(const mach_header*, intptr_t) {
    imagesRemoved.fetch_add(1, std::memory_order_relaxed);
}

__attribute__((constructor)) void TrackImages() {
    _dyld_register_func_for_add_image(OnImageAdded);
    _dyld_register_func_for_remove_image(OnImageRemoved);
}

}

// The arm64 runner maps the executable and the guest modules itself. They join dl_iterate_phdr, and the
// guest memory registry right away, whether or not it has registered the dyld images yet, so the guest can
// query and protect their pages.
extern "C" void Aps5RegisterGuestImage_nid_no_patch(const dl_phdr_info* image) {
    GuestAllocations::RegisterRunnerImage(*image);
    auto& registered = Registered();
    std::lock_guard lock(registered.mutex);
    registered.images.push_back({image->dlpi_addr, image->dlpi_name, image->dlpi_phdr, image->dlpi_phnum});
    imagesAdded.fetch_add(1, std::memory_order_relaxed);
}

#ifdef __aarch64__
namespace {
std::atomic<void* (*)(const void*, const char*)> bridgeCallable {nullptr};
}

extern "C" void Aps5SetBridgeCallable_nid_no_patch(void* (*callable)(const void* address, const char* name)) {
    bridgeCallable.store(callable, std::memory_order_release);
}

extern "C" void* LibcGuestCallable_nid_no_patch(const void* address, const char* name) {
    const auto callable = bridgeCallable.load(std::memory_order_acquire);
    return callable != nullptr && address != nullptr ? callable(address, name) : const_cast<void*>(address);
}
#endif

extern "C" int dl_iterate_phdr(int (*callback)(dl_phdr_info*, std::size_t, void*), void* data) {
    std::vector<dl_phdr_info> images;
    {
        auto& registered = Registered();
        std::lock_guard lock(registered.mutex);
        images = registered.images;
    }
    for (auto& image : images) {
        image.dlpi_adds = imagesAdded.load(std::memory_order_relaxed);
        image.dlpi_subs = imagesRemoved.load(std::memory_order_relaxed);
        if (const int result = callback(&image, sizeof(image), data); result != 0) return result;
    }
    for (std::uint32_t index = 0; index < _dyld_image_count(); ++index) {
        const auto* header = reinterpret_cast<const mach_header_64*>(_dyld_get_image_header(index));
        if (header == nullptr || header->magic != MH_MAGIC_64 || (header->flags & MH_DYLIB_IN_CACHE) != 0) continue;
        unsigned long size = 0;
        const auto* meta = getsectiondata(header, "__APS5DATA", "__meta", &size);
        if (meta == nullptr || size < HeadersOffset) continue;
        std::uint64_t version;
        std::uint64_t loadBias;
        std::uint64_t count;
        std::memcpy(&version, meta, sizeof(version));
        std::memcpy(&loadBias, meta + LoadBiasOffset, sizeof(loadBias));
        std::memcpy(&count, meta + HeaderCountOffset, sizeof(count));
        if (version != MetaVersion || count > 0xffff || count > (size - HeadersOffset) / sizeof(Elf64_Phdr)) continue;
        dl_phdr_info info {};
        info.dlpi_addr = static_cast<std::uintptr_t>(loadBias);
        info.dlpi_name = header->filetype == MH_EXECUTE ? "" : _dyld_get_image_name(index);
        info.dlpi_phdr = reinterpret_cast<const Elf64_Phdr*>(meta + HeadersOffset);
        info.dlpi_phnum = static_cast<std::uint16_t>(count);
        info.dlpi_adds = imagesAdded.load(std::memory_order_relaxed);
        info.dlpi_subs = imagesRemoved.load(std::memory_order_relaxed);
        if (const int result = callback(&info, sizeof(info), data); result != 0) return result;
    }
    return 0;
}

#endif
