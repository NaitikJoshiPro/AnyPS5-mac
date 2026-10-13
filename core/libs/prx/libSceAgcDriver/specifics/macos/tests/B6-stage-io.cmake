# Work package B6 (stage interfaces): Metal-only tests. Included by Metal.cmake when BUILD_TESTING is on.
# agc_metal_msl_stage_io: vertex/fragment functions emitted for seeded recompile requests, checked on the CPU.
add_test_executable(agc_metal_msl_stage_io_tests ${agcMetalDir}/tests/MslStageIoCheck.cpp)
target_include_directories(agc_metal_msl_stage_io_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/core/shader/recompiler $<TARGET_PROPERTY:shader_recompiler_agc_driver,INTERFACE_INCLUDE_DIRECTORIES> ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR} ${ANYPS5_SPIRV_HEADERS_INCLUDE_DIR})
target_link_libraries(agc_metal_msl_stage_io_tests PRIVATE libSceAgcDriver SDL2-static libc)
add_test(NAME agc_metal_msl_stage_io COMMAND agc_metal_msl_stage_io_tests ${agcMetalDir}/tests/msl-stage-io)
set_tests_properties(agc_metal_msl_stage_io PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 60)
# agc_metal_msl_per_vertex: emulated per-vertex fragment values (MslAbi Version 5), compiled and linked
# in the triangle-vertices mode with Metal (compile only, no GPU work).
add_test_executable(agc_metal_msl_per_vertex_tests ${agcMetalDir}/tests/MslPerVertexCheck.mm)
target_include_directories(agc_metal_msl_per_vertex_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/core/shader/recompiler $<TARGET_PROPERTY:shader_recompiler_agc_driver,INTERFACE_INCLUDE_DIRECTORIES> ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR} ${ANYPS5_SPIRV_HEADERS_INCLUDE_DIR})
target_link_libraries(agc_metal_msl_per_vertex_tests PRIVATE libSceAgcDriver SDL2-static libc "-framework Metal" "-framework Foundation")
set_source_files_properties(${agcMetalDir}/tests/MslPerVertexCheck.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
set_target_properties(agc_metal_msl_per_vertex_tests PROPERTIES CXX_EXTENSIONS OFF OBJCXX_STANDARD 20)
if(CMAKE_CXX_COMPILER_LAUNCHER)
    set_property(TARGET agc_metal_msl_per_vertex_tests PROPERTY OBJCXX_COMPILER_LAUNCHER ${CMAKE_CXX_COMPILER_LAUNCHER})
endif()
add_test(NAME agc_metal_msl_per_vertex COMMAND agc_metal_msl_per_vertex_tests)
set_tests_properties(agc_metal_msl_per_vertex PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1" SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 60)
