# Work package D5 (descriptors, samplers, pipelines, caches): Metal-only tests. Included by Metal.cmake when BUILD_TESTING is on.
# agc_metal_pipeline_cache: two processes share an archive directory; the second compiles 0 pipelines (MISSION §8).
# agc_metal_pipeline_cache_background: background preparation, concurrent requests, failures, cache keys.
add_test_executable(agc_metal_pipeline_cache_tests ${agcMetalDir}/tests/MetalPipelineCacheTests.mm)
target_include_directories(agc_metal_pipeline_cache_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/core/shader/recompiler $<TARGET_PROPERTY:shader_recompiler_agc_driver,INTERFACE_INCLUDE_DIRECTORIES> ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR} ${ANYPS5_SPIRV_HEADERS_INCLUDE_DIR})
target_link_libraries(agc_metal_pipeline_cache_tests PRIVATE libSceAgcDriver libc "-framework Metal" "-framework Foundation")
set_source_files_properties(${agcMetalDir}/tests/MetalPipelineCacheTests.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
set_target_properties(agc_metal_pipeline_cache_tests PROPERTIES CXX_EXTENSIONS OFF OBJCXX_STANDARD 20)
if(CMAKE_CXX_COMPILER_LAUNCHER)
    set_property(TARGET agc_metal_pipeline_cache_tests PROPERTY OBJCXX_COMPILER_LAUNCHER ${CMAKE_CXX_COMPILER_LAUNCHER})
endif()
add_test(NAME agc_metal_pipeline_cache COMMAND agc_metal_pipeline_cache_tests)
add_test(NAME agc_metal_pipeline_cache_background COMMAND agc_metal_pipeline_cache_tests --background)
# agc_metal_pipeline_cache_save_race: archive saves while other threads compile (the archive is off under validation).
add_test(NAME agc_metal_pipeline_cache_save_race COMMAND agc_metal_pipeline_cache_tests --save-race)
set_tests_properties(agc_metal_pipeline_cache agc_metal_pipeline_cache_background agc_metal_pipeline_cache_save_race PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
set_tests_properties(agc_metal_pipeline_cache_background PROPERTIES ENVIRONMENT "APS5_NO_PIPELINE_CACHE=1")

# agc_metal_library_store (DECISIONS D55): the persistent MSL library store and the startup prewarm.
add_test_executable(agc_metal_library_store_tests ${agcMetalDir}/tests/MetalLibraryStoreTests.mm)
target_include_directories(agc_metal_library_store_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/core/shader/recompiler $<TARGET_PROPERTY:shader_recompiler_agc_driver,INTERFACE_INCLUDE_DIRECTORIES> ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR} ${ANYPS5_SPIRV_HEADERS_INCLUDE_DIR})
target_link_libraries(agc_metal_library_store_tests PRIVATE libSceAgcDriver libc "-framework Metal" "-framework Foundation")
set_source_files_properties(${agcMetalDir}/tests/MetalLibraryStoreTests.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
set_target_properties(agc_metal_library_store_tests PROPERTIES CXX_EXTENSIONS OFF OBJCXX_STANDARD 20)
if(CMAKE_CXX_COMPILER_LAUNCHER)
    set_property(TARGET agc_metal_library_store_tests PROPERTY OBJCXX_COMPILER_LAUNCHER ${CMAKE_CXX_COMPILER_LAUNCHER})
endif()
add_test(NAME agc_metal_library_store COMMAND agc_metal_library_store_tests)
set_tests_properties(agc_metal_library_store PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
