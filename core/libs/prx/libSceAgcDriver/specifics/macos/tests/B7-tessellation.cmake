# Tessellation on Metal (DECISIONS D24): Metal-only tests. Included by Metal.cmake when BUILD_TESTING is on.
# agc_metal_tessellator: the software tessellator the tessellation mesh functions run, checked on the CPU
# against Microsoft's D3D11 reference tessellator (3rdparty/d3d11-tessellator, MIT). No Metal device.
add_test_executable(agc_metal_tessellator_tests ${agcMetalDir}/tests/TessellatorCheck.cpp ${CMAKE_SOURCE_DIR}/3rdparty/d3d11-tessellator/tessellator.cpp)
target_include_directories(agc_metal_tessellator_tests PRIVATE ${CMAKE_SOURCE_DIR}/core/shader/recompiler/MetalBackend/include ${CMAKE_SOURCE_DIR}/3rdparty/d3d11-tessellator)
set_target_properties(agc_metal_tessellator_tests PROPERTIES CXX_STANDARD 20 CXX_EXTENSIONS OFF)
set_source_files_properties(${CMAKE_SOURCE_DIR}/3rdparty/d3d11-tessellator/tessellator.cpp PROPERTIES COMPILE_OPTIONS "-w")
add_test(NAME agc_metal_tessellator COMMAND agc_metal_tessellator_tests)
set_tests_properties(agc_metal_tessellator PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 120)
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
