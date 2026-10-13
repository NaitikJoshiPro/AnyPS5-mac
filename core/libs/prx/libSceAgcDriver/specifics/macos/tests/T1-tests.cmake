# Work package T1 (test infrastructure): Metal-only tests. Included by Metal.cmake when BUILD_TESTING is on.
# agc_metal_request_dump checks the APS5_DUMP_REQUESTS hook; agc_metal_msl_corpus replays the recompile requests a
# GPU suite run dumped into build-metal/msl-corpus (tools/mac/gpu_suite.sh all --dump-requests build-metal/msl-corpus)
# and the committed seed corpus through MetalBackend and the Metal compiler (M2 exit criterion 1).
foreach(agcMetalT1Test IN ITEMS msl_corpus request_dump)
    if(agcMetalT1Test STREQUAL "msl_corpus")
        set(agcMetalT1Source ${agcMetalDir}/tests/MslCorpus.mm)
    else()
        set(agcMetalT1Source ${agcMetalDir}/tests/RequestDump.mm)
    endif()
    add_test_executable(agc_metal_${agcMetalT1Test}_tests ${agcMetalT1Source})
    target_include_directories(agc_metal_${agcMetalT1Test}_tests PRIVATE
            ${LIBS_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/core/shader/recompiler
            $<TARGET_PROPERTY:shader_recompiler_agc_driver,INTERFACE_INCLUDE_DIRECTORIES>
            ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR}
            ${ANYPS5_SPIRV_HEADERS_INCLUDE_DIR}
    )
    target_link_libraries(agc_metal_${agcMetalT1Test}_tests PRIVATE libSceAgcDriver SDL2-static libc "-framework Metal" "-framework Foundation")
    set_source_files_properties(${agcMetalT1Source} PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    set_target_properties(agc_metal_${agcMetalT1Test}_tests PROPERTIES CXX_EXTENSIONS OFF OBJCXX_STANDARD 20)
    if(CMAKE_CXX_COMPILER_LAUNCHER)
        set_property(TARGET agc_metal_${agcMetalT1Test}_tests PROPERTY OBJCXX_COMPILER_LAUNCHER ${CMAKE_CXX_COMPILER_LAUNCHER})
    endif()
endforeach()
add_test(NAME agc_metal_msl_corpus COMMAND agc_metal_msl_corpus_tests --corpus ${CMAKE_BINARY_DIR}/msl-corpus --corpus ${agcMetalDir}/tests/msl-corpus)
set_tests_properties(agc_metal_msl_corpus PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1" SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 900)
add_test(NAME agc_metal_request_dump COMMAND agc_metal_request_dump_tests)
set_tests_properties(agc_metal_request_dump PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1" SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 60)

# agc_descriptor_pool_reset_failure (tests/Graphics.cpp --descriptor-pool-reset-failure) skips on Metal: descriptor
# pools are tokens whose reset cannot fail (GraphicsTwin.mm; allowlisted, TechnicalDebt metal-vulkan-internal-tests).
if(TEST agc_descriptor_pool_reset_failure)
    set_tests_properties(agc_descriptor_pool_reset_failure PROPERTIES SKIP_RETURN_CODE 77)
endif()
