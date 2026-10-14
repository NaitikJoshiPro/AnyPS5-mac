#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DISPLAYFORMAT_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DISPLAYFORMAT_HPP

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <cstdint>

namespace AgcDriver {

VkFormat DisplayTexelFormat(std::uint64_t pixelFormat);
bool DisplayTenBit(std::uint64_t pixelFormat);
bool DisplayRedLow(std::uint64_t pixelFormat);
bool DisplayHalfFloat(std::uint64_t pixelFormat);
std::uint32_t DisplayTexelBytes(std::uint64_t pixelFormat);

enum class ResidentPresent : std::uint8_t { None, Blit, Convert };
ResidentPresent ResidentPresentPath(VkFormat storage, std::uint64_t pixelFormat, bool blitSource);
// A presenter that samples the image holding a display buffer's pending results (Metal, D54) shows
// 8-bit UNORM images only, in the display's channel order or with red and blue exchanged. sRGB
// storage (sampling would decode it), 10-bit images (sampling rounds where the guest path truncates)
// and half floats go through guest memory.
enum class ResidentSample : std::uint8_t { None, Direct, SwapRedBlue };
ResidentSample ResidentSamplePath(VkFormat storage, std::uint64_t pixelFormat);
// The layer of a single-mip image (`layers` layers over `imageBytes` from `imageBase`) a display
// buffer of `bytes` at `address` is: a swap chain is often one array image whose layers are the
// display buffers, contiguous and in order. -1 when the buffer is not exactly one layer.
std::int64_t ResidentLayer(std::uint64_t imageBase, std::uint64_t imageBytes, std::uint32_t layers, std::uint64_t address, std::uint64_t bytes);

}

#endif
