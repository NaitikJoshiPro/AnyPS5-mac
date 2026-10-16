# Work package B3 (wave, LDS, atomics): Metal-only tests. Included by Metal.cmake when BUILD_TESTING is on.
# agc_metal_msl_wave_lds: the MSL emitted for wave/LDS/64-bit atomic programs, checked on the CPU (no Metal device).
add_test_executable(agc_metal_msl_wave_lds_tests ${agcMetalDir}/tests/MslWaveLds.cpp)
target_include_directories(agc_metal_msl_wave_lds_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/core/shader/recompiler $<TARGET_PROPERTY:shader_recompiler_agc_driver,INTERFACE_INCLUDE_DIRECTORIES> ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR} ${ANYPS5_SPIRV_HEADERS_INCLUDE_DIR})
target_link_libraries(agc_metal_msl_wave_lds_tests PRIVATE libSceAgcDriver SDL2-static libc)
add_test(NAME agc_metal_msl_wave_lds COMMAND agc_metal_msl_wave_lds_tests)
set_tests_properties(agc_metal_msl_wave_lds PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 60)
# agc_metal_wave64_lane_local_plan (DECISIONS D62): the wave64 lane-local layout decision on the CPU (no Metal device).
add_test_executable(agc_metal_wave64_lane_local_plan_tests ${agcMetalDir}/tests/Wave64LaneLocalPlan.cpp)
target_include_directories(agc_metal_wave64_lane_local_plan_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/core/shader/recompiler $<TARGET_PROPERTY:shader_recompiler_agc_driver,INTERFACE_INCLUDE_DIRECTORIES> ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR} ${ANYPS5_SPIRV_HEADERS_INCLUDE_DIR})
target_link_libraries(agc_metal_wave64_lane_local_plan_tests PRIVATE libSceAgcDriver SDL2-static libc)
add_test(NAME agc_metal_wave64_lane_local_plan COMMAND agc_metal_wave64_lane_local_plan_tests)
set_tests_properties(agc_metal_wave64_lane_local_plan PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1;APS5_MSL_LANE_SURVEY=${CMAKE_BINARY_DIR}/wave64-lane-local-plan.txt" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 60)
