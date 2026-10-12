# Work package D2 (recorder and buffer primitives): Metal-only tests. Included by Metal.cmake when BUILD_TESTING is on.
# The Vulkan-mock tests of the recorder and the buffer pool (agc_recorder_kept_bytes, agc_buffer_pool) keep their
# names and registrations: their sources are swapped for Metal twins in twins/D2-recorder.cmake.
add_test_executable(agc_metal_recorder_tests ${agcMetalDir}/tests/MetalRecorder.mm)
target_include_directories(agc_metal_recorder_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/core/shader/recompiler ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR})
target_link_libraries(agc_metal_recorder_tests PRIVATE libSceAgcDriver libc "-framework Metal" "-framework Foundation")
set_source_files_properties(${agcMetalDir}/tests/MetalRecorder.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
set_target_properties(agc_metal_recorder_tests PROPERTIES CXX_EXTENSIONS OFF OBJCXX_STANDARD 20)
if(CMAKE_CXX_COMPILER_LAUNCHER)
    set_property(TARGET agc_metal_recorder_tests PROPERTY OBJCXX_COMPILER_LAUNCHER ${CMAKE_CXX_COMPILER_LAUNCHER})
endif()
add_test(NAME agc_metal_recorder COMMAND agc_metal_recorder_tests)
add_test(NAME agc_metal_recorder_timing COMMAND agc_metal_recorder_tests --timing)
set_tests_properties(agc_metal_recorder agc_metal_recorder_timing PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
set_tests_properties(agc_metal_recorder_timing PROPERTIES ENVIRONMENT "APS5_PROFILE_GPU=1")
