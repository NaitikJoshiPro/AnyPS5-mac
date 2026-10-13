# Tessellation on Metal (DECISIONS D24): Metal-only tests. Included by Metal.cmake when BUILD_TESTING is on.
# agc_metal_tessellator: the software tessellator the tessellation mesh functions run, checked on the CPU
# against Microsoft's D3D11 reference tessellator (3rdparty/d3d11-tessellator, MIT). No Metal device.
add_test_executable(agc_metal_tessellator_tests ${agcMetalDir}/tests/TessellatorCheck.cpp ${CMAKE_SOURCE_DIR}/3rdparty/d3d11-tessellator/tessellator.cpp)
target_include_directories(agc_metal_tessellator_tests PRIVATE ${CMAKE_SOURCE_DIR}/core/shader/recompiler/MetalBackend/include ${CMAKE_SOURCE_DIR}/3rdparty/d3d11-tessellator)
set_target_properties(agc_metal_tessellator_tests PROPERTIES CXX_STANDARD 20 CXX_EXTENSIONS OFF)
set_source_files_properties(${CMAKE_SOURCE_DIR}/3rdparty/d3d11-tessellator/tessellator.cpp PROPERTIES COMPILE_OPTIONS "-w")
add_test(NAME agc_metal_tessellator COMMAND agc_metal_tessellator_tests)
set_tests_properties(agc_metal_tessellator PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 120)
