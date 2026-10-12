# Work package B6 (stage interfaces): Metal-only tests. Included by Metal.cmake when BUILD_TESTING is on.
# agc_metal_msl_stage_io: vertex/fragment functions emitted for seeded recompile requests, checked on the CPU.
add_test_executable(agc_metal_msl_stage_io_tests ${agcMetalDir}/tests/MslStageIoCheck.cpp)
target_include_directories(agc_metal_msl_stage_io_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/core/shader/recompiler $<TARGET_PROPERTY:shader_recompiler_agc_driver,INTERFACE_INCLUDE_DIRECTORIES> ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR} ${ANYPS5_SPIRV_HEADERS_INCLUDE_DIR})
target_link_libraries(agc_metal_msl_stage_io_tests PRIVATE libSceAgcDriver SDL2-static libc)
add_test(NAME agc_metal_msl_stage_io COMMAND agc_metal_msl_stage_io_tests ${agcMetalDir}/tests/msl-stage-io)
set_tests_properties(agc_metal_msl_stage_io PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 60)
