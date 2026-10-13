# D3D11 reference tessellator (test oracle)

`tessellator.cpp` / `tessellator.hpp`: Microsoft's D3D11 reference tessellator (CHWTessellator),
Copyright (c) Microsoft Corporation, MIT licence (header of each file), unmodified copies of Mesa's
`src/gallium/auxiliary/tessellator/` at commit 65c2dc5a634b6ee31979546ec5eb5212d6ffe721
(sha256 tessellator.cpp c5c986fe2c31ccb674e6c980895d5adc93cd1d24990d9354b5e63a876355db36,
tessellator.hpp 14fcefcf31d9afbbd894e088f56b0c77f7c235f1fbe8a5043b295f7e05cecb8d).
`util/macros.h` stands in for Mesa's header (only UNREACHABLE).

Used only by the `agc_metal_tessellator` test, which checks the Metal path's software tessellator
(core/shader/recompiler/MetalBackend/include/MetalBackend/MslTessellatorCore.inc, a port of the same
algorithm) against it point for point and index for index. Recorded in docs/mac/DECISIONS.md D24.
