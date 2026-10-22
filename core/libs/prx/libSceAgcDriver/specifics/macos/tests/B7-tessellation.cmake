# Tessellation on Metal (DECISIONS D24): Metal-only tests. Included by Metal.cmake when BUILD_TESTING is on.
# agc_metal_tessellator: the software tessellator the tessellation mesh functions run, checked on the CPU
# against Microsoft's D3D11 reference tessellator (3rdparty/d3d11-tessellator, MIT). No Metal device.
add_test_executable(agc_metal_tessellator_tests ${agcMetalDir}/tests/TessellatorCheck.cpp ${CMAKE_SOURCE_DIR}/3rdparty/d3d11-tessellator/tessellator.cpp)
target_include_directories(agc_metal_tessellator_tests PRIVATE ${CMAKE_SOURCE_DIR}/core/shader/recompiler/MetalBackend/include ${CMAKE_SOURCE_DIR}/3rdparty/d3d11-tessellator)
set_target_properties(agc_metal_tessellator_tests PROPERTIES CXX_STANDARD 20 CXX_EXTENSIONS OFF)
set_source_files_properties(${CMAKE_SOURCE_DIR}/3rdparty/d3d11-tessellator/tessellator.cpp PROPERTIES COMPILE_OPTIONS "-w")
add_test(NAME agc_metal_tessellator COMMAND agc_metal_tessellator_tests)
set_tests_properties(agc_metal_tessellator PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 120)
# agc_driver_tessellation runs through TessellationTwin.mm (twins/UPSYNC3-tests.cmake), which links the stages
# with LinkMslTessellation to check the partitioning upstream reads from the SPIR-V.
if(TARGET agc_driver_tessellation_tests)
    target_include_directories(agc_driver_tessellation_tests PRIVATE ${CMAKE_SOURCE_DIR}/core/shader/recompiler/MetalBackend/include $<TARGET_PROPERTY:shader_recompiler_agc_driver,INTERFACE_INCLUDE_DIRECTORIES>)
endif()
# agc_metal_msl_tessellation: tests/execution/Tessellation.cpp's local, hull and domain programs as MSL
# tessellation libraries, linked (LinkMslTessellation) into object and mesh functions for every
# domain, partitioning and topology and into mesh render pipelines with the pixel program (compile
# only, no GPU work).
add_test_executable(agc_metal_msl_tessellation_tests ${agcMetalDir}/tests/MslTessellationCheck.mm)
target_include_directories(agc_metal_msl_tessellation_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/core/shader/recompiler ${CMAKE_SOURCE_DIR}/core/shader/recompiler/MetalBackend/include $<TARGET_PROPERTY:shader_recompiler_agc_driver,INTERFACE_INCLUDE_DIRECTORIES> ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR} ${ANYPS5_SPIRV_HEADERS_INCLUDE_DIR})
target_link_libraries(agc_metal_msl_tessellation_tests PRIVATE libSceAgcDriver SDL2-static libc "-framework Metal" "-framework Foundation")
set_source_files_properties(${agcMetalDir}/tests/MslTessellationCheck.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
set_target_properties(agc_metal_msl_tessellation_tests PROPERTIES CXX_EXTENSIONS OFF OBJCXX_STANDARD 20)
if(CMAKE_CXX_COMPILER_LAUNCHER)
    set_property(TARGET agc_metal_msl_tessellation_tests PROPERTY OBJCXX_COMPILER_LAUNCHER ${CMAKE_CXX_COMPILER_LAUNCHER})
endif()
add_test(NAME agc_metal_msl_tessellation COMMAND agc_metal_msl_tessellation_tests)
set_tests_properties(agc_metal_msl_tessellation PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1" SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 60)
# agc_metal_tessellation_domains (DECISIONS D26): guest tessellated draws of quad, triangle and isoline
# patches with every partitioning and the point, line, cw and ccw outputs, culled patches and level
# clamps, checked against the software tessellator core (domain program runs per unique slice point,
# its (u, v) bit for bit, coverage and winding of the rendered patch). GPU, one patch per draw.
add_test_executable(agc_metal_tessellation_domains_tests ${agcMetalDir}/tests/TessellationDomains.cpp)
target_include_directories(agc_metal_tessellation_domains_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/core/shader/recompiler ${CMAKE_SOURCE_DIR}/core/shader/recompiler/MetalBackend/include ${CMAKE_SOURCE_DIR}/core/libs/prx/libSceAgcDriver/tests/execution ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR} ${ANYPS5_SPIRV_HEADERS_INCLUDE_DIR})
target_link_libraries(agc_metal_tessellation_domains_tests PRIVATE libSceAgcDriver SDL2-static libc)
set_target_properties(agc_metal_tessellation_domains_tests PROPERTIES CXX_STANDARD 20 CXX_EXTENSIONS OFF)
add_test(NAME agc_metal_tessellation_domains COMMAND agc_metal_tessellation_domains_tests)
set_tests_properties(agc_metal_tessellation_domains PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1" SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 60 LABELS "gpu")
# agc_metal_tessellator_gpu (DECISIONS D26): the tessellator core compiled as MSL and run on the GPU
# (one thread per factor set) equals the same core compiled as C++: counts, slice windows, points.
add_test_executable(agc_metal_tessellator_gpu_tests ${agcMetalDir}/tests/TessellatorGpuCheck.mm)
target_include_directories(agc_metal_tessellator_gpu_tests PRIVATE ${CMAKE_SOURCE_DIR}/core/shader/recompiler/MetalBackend/include)
target_link_libraries(agc_metal_tessellator_gpu_tests PRIVATE "-framework Metal" "-framework Foundation")
set_source_files_properties(${agcMetalDir}/tests/TessellatorGpuCheck.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
set_target_properties(agc_metal_tessellator_gpu_tests PROPERTIES CXX_EXTENSIONS OFF OBJCXX_STANDARD 20)
if(CMAKE_CXX_COMPILER_LAUNCHER)
    set_property(TARGET agc_metal_tessellator_gpu_tests PROPERTY OBJCXX_COMPILER_LAUNCHER ${CMAKE_CXX_COMPILER_LAUNCHER})
endif()
add_test(NAME agc_metal_tessellator_gpu COMMAND agc_metal_tessellator_gpu_tests)
set_tests_properties(agc_metal_tessellator_gpu PROPERTIES SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 30 LABELS "gpu")
# agc_metal_subword_threadgroup: minimal repro of the TESS2 finding (83ed2d518): the pre-fix vertex
# sharing shape with uchar/ushort threadgroup arrays next to an int array, in a compute kernel and a
# mesh function, against a 32-bit-word control. Records whether the corruption reproduces in
# isolation (the emitters declare only 32-bit threadgroup arrays). GPU, one 256-thread threadgroup.
add_test_executable(agc_metal_subword_threadgroup_tests ${agcMetalDir}/tests/SubwordThreadgroupCheck.mm)
target_link_libraries(agc_metal_subword_threadgroup_tests PRIVATE "-framework Metal" "-framework Foundation")
set_source_files_properties(${agcMetalDir}/tests/SubwordThreadgroupCheck.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
set_target_properties(agc_metal_subword_threadgroup_tests PROPERTIES CXX_EXTENSIONS OFF OBJCXX_STANDARD 20)
if(CMAKE_CXX_COMPILER_LAUNCHER)
    set_property(TARGET agc_metal_subword_threadgroup_tests PROPERTY OBJCXX_COMPILER_LAUNCHER ${CMAKE_CXX_COMPILER_LAUNCHER})
endif()
add_test(NAME agc_metal_subword_threadgroup COMMAND agc_metal_subword_threadgroup_tests)
set_tests_properties(agc_metal_subword_threadgroup PROPERTIES SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 30 LABELS "gpu")
