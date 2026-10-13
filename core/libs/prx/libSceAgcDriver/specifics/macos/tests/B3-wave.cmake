# Work package B3 (wave, LDS, atomics): Metal-only tests. Included by Metal.cmake when BUILD_TESTING is on.
# agc_metal_msl_wave_lds: the MSL emitted for wave/LDS/64-bit atomic programs, checked on the CPU (no Metal device).
add_test_executable(agc_metal_msl_wave_lds_tests ${agcMetalDir}/tests/MslWaveLds.cpp)
target_include_directories(agc_metal_msl_wave_lds_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/core/shader/recompiler $<TARGET_PROPERTY:shader_recompiler_agc_driver,INTERFACE_INCLUDE_DIRECTORIES> ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR} ${ANYPS5_SPIRV_HEADERS_INCLUDE_DIR})
target_link_libraries(agc_metal_msl_wave_lds_tests PRIVATE libSceAgcDriver SDL2-static libc)
add_test(NAME agc_metal_msl_wave_lds COMMAND agc_metal_msl_wave_lds_tests)
set_tests_properties(agc_metal_msl_wave_lds PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 60)
