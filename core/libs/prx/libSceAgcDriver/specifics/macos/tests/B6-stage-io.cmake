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
# agc_metal_msl_mesh: MSL mesh functions (NGG primitive shaders) of tests/Mesh.cpp's geometry program,
# compiled and linked with the fragment function into mesh render pipelines (compile only, no GPU work).
add_test_executable(agc_metal_msl_mesh_tests ${agcMetalDir}/tests/MslMeshCheck.mm)
target_include_directories(agc_metal_msl_mesh_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/core/shader/recompiler $<TARGET_PROPERTY:shader_recompiler_agc_driver,INTERFACE_INCLUDE_DIRECTORIES> ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR} ${ANYPS5_SPIRV_HEADERS_INCLUDE_DIR})
target_link_libraries(agc_metal_msl_mesh_tests PRIVATE libSceAgcDriver SDL2-static libc "-framework Metal" "-framework Foundation")
set_source_files_properties(${agcMetalDir}/tests/MslMeshCheck.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
set_target_properties(agc_metal_msl_mesh_tests PROPERTIES CXX_EXTENSIONS OFF OBJCXX_STANDARD 20)
if(CMAKE_CXX_COMPILER_LAUNCHER)
    set_property(TARGET agc_metal_msl_mesh_tests PROPERTY OBJCXX_COMPILER_LAUNCHER ${CMAKE_CXX_COMPILER_LAUNCHER})
endif()
add_test(NAME agc_metal_msl_mesh COMMAND agc_metal_msl_mesh_tests)
set_tests_properties(agc_metal_msl_mesh PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1" SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 60)
# agc_metal_msl_upstream_parity (MslAbi Versions 10 and 11): the upstream syncs' SPIR-V backend changes
# as the MSL backend mirrors them (flat P10, emulated cull distances, one-lane vertex waves, hardware f16
# conversion; compared f16 operands, early fragment tests, the ancillary layer field, coherent GDS),
# compiled with Metal (compile and pipeline link only, no GPU work).
add_test_executable(agc_metal_msl_upstream_parity_tests ${agcMetalDir}/tests/MslUpstreamParity.mm)
target_include_directories(agc_metal_msl_upstream_parity_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/core/shader/recompiler $<TARGET_PROPERTY:shader_recompiler_agc_driver,INTERFACE_INCLUDE_DIRECTORIES> ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR} ${ANYPS5_SPIRV_HEADERS_INCLUDE_DIR})
target_link_libraries(agc_metal_msl_upstream_parity_tests PRIVATE libSceAgcDriver SDL2-static libc "-framework Metal" "-framework Foundation")
set_source_files_properties(${agcMetalDir}/tests/MslUpstreamParity.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
set_target_properties(agc_metal_msl_upstream_parity_tests PROPERTIES CXX_EXTENSIONS OFF OBJCXX_STANDARD 20)
if(CMAKE_CXX_COMPILER_LAUNCHER)
    set_property(TARGET agc_metal_msl_upstream_parity_tests PROPERTY OBJCXX_COMPILER_LAUNCHER ${CMAKE_CXX_COMPILER_LAUNCHER})
endif()
add_test(NAME agc_metal_msl_upstream_parity COMMAND agc_metal_msl_upstream_parity_tests)
set_tests_properties(agc_metal_msl_upstream_parity PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1" SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 60)
# agc_metal_msl_offset_rate: whether a fragment function reading interpolant<>::interpolate_at_offset runs
# once per pixel like a center_perspective one (OffsetBarycentric in once-per-pixel programs), counted in a
# 2x2, 4x attachment-less pass. GPU, 4 tiny draws.
add_test_executable(agc_metal_msl_offset_rate_tests ${agcMetalDir}/tests/MslOffsetRateCheck.mm)
target_link_libraries(agc_metal_msl_offset_rate_tests PRIVATE "-framework Metal" "-framework Foundation")
set_source_files_properties(${agcMetalDir}/tests/MslOffsetRateCheck.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
set_target_properties(agc_metal_msl_offset_rate_tests PROPERTIES CXX_EXTENSIONS OFF OBJCXX_STANDARD 20)
if(CMAKE_CXX_COMPILER_LAUNCHER)
    set_property(TARGET agc_metal_msl_offset_rate_tests PROPERTY OBJCXX_COMPILER_LAUNCHER ${CMAKE_CXX_COMPILER_LAUNCHER})
endif()
add_test(NAME agc_metal_msl_offset_rate COMMAND agc_metal_msl_offset_rate_tests)
set_tests_properties(agc_metal_msl_offset_rate PROPERTIES SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 30 LABELS "gpu")
