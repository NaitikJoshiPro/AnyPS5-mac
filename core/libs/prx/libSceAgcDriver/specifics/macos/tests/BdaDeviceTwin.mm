// Metal twin of tests/BdaDevice.cpp (agc_driver_bda_device), swapped in by twins/T1-tests.cmake. The BDA contract
// checks are CPU-side and run as upstream; the device part allocates Vulkan buffers with device addresses and runs
// SPIR-V BDA shaders, whose Metal port (the BDA page table over wrapped guest memory and MSL BDA access) belongs to
// work packages D3 and B5. Until it is registered here, the test skips naming them.

#include "../../../tests/BdaShader.hpp"
#include <cstdio>
#include <exception>
#include <iostream>

int main() {
    try {
        RunBdaContractTests();
        std::puts("BDA contract tests passed");
        std::puts("skipped, the Metal twin of the BDA device allocation and execution checks (agc_driver_bda_device) is not written yet: work packages D3 (BDA page table) and B5 (MSL BDA access)");
        return 77;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
