// Twin of Graphics/src/BufferAddress.cpp (work package D2): a Metal buffer's GPU address is known
// from its creation (the pool records it), so nothing is resolved here.

#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"

namespace AgcDriver::Graphics {

VkDeviceAddress Buffer::DeviceAddress() const {
    Require(deviceAddress != 0, "buffer has no device address");
    return deviceAddress;
}

}
