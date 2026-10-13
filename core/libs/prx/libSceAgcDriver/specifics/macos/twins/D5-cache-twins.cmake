# Work package D5 (descriptor tables, texture and resource caches): upstream execution tests that build a Vulkan
# device = Metal twins (docs/mac/M2_DESIGN.md §8). Each compiles the upstream file unchanged with its main renamed
# and runs its checks on the Metal image context: OversizedDescriptorSetsTwin.mm (DescriptorCache's dedicated pools
# and a dispatch through a 4097-entry MslAbi descriptor table), SampledCacheBudgetTwin.mm (sampled texture cache
# budget, memory budget and the resource cache admission/clear modes).
list(APPEND agcMetalTwins
        "tests/execution/OversizedDescriptorSets.cpp=specifics/macos/tests/OversizedDescriptorSetsTwin.mm"
        "tests/execution/SampledCacheBudget.cpp=specifics/macos/tests/SampledCacheBudgetTwin.mm"
)
# The oversized-table kernel is built from the MSL backend's prelude and table declaration (MetalBackend headers).
if(TARGET agc_driver_oversized_descriptor_sets_tests)
    target_include_directories(agc_driver_oversized_descriptor_sets_tests PRIVATE $<TARGET_PROPERTY:shader_recompiler_agc_driver,INTERFACE_INCLUDE_DIRECTORIES>)
endif()
