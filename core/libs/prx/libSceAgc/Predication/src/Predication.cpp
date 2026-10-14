#include "prx/libSceAgc/Predication/include/Predication.hpp"

#include "prx/libSceAgc/Command/include/Packet.hpp"

#include <cstdint>
#include <cstddef>
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"

extern "C" {

int APS5_VABI sceAgcSetPacketPredication(uint32_t* packet, uint32_t predication) {
    Agc::Command::Require(packet != nullptr, __func__, "null packet");
    Agc::Command::CheckBits(predication, 1, __func__);
    packet = Agc::Command::Host(packet);
    Agc::Command::Require((packet[0] >> 30u) == 3u, __func__, "packet is not a type-3 packet");
    packet[0] = (packet[0] & ~1u) | predication;
    return 0;
}

int APS5_VABI sceAgcSetRangePredication(uint32_t* start, const volatile uint32_t* end, uint32_t predication) {
    Agc::Command::Require(start != nullptr && end != nullptr && start <= end, __func__, "invalid packet range");
    Agc::Command::CheckBits(predication, 1, __func__);
    // One mapping holds the range: walk its host view.
    auto* packet = Agc::Command::Host(start);
    const auto* const hostEnd = packet + (end - start);
    while (packet < hostEnd) {
        const auto type = packet[0] >> 30u;
        if (type == 2u || (packet[0] & 0x3fffff00u) == 0x3fff1000u) {
            ++packet;
            continue;
        }
        Agc::Command::Require(type == 3u, __func__, "range holds a packet that is neither type 2 nor type 3");
        const auto dwords = ((packet[0] >> 16u) & 0x3fffu) + 2u;
        Agc::Command::Require(static_cast<std::size_t>(hostEnd - packet) >= dwords, __func__, "range ends inside a packet");
        packet[0] = (packet[0] & ~1u) | predication;
        packet += dwords;
    }
    return 0;
}

}
