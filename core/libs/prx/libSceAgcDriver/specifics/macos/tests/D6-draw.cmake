# Work package D6 (draws, attachments, presentation): Metal-only tests. Included by Metal.cmake when BUILD_TESTING is on.
add_test_executable(agc_metal_draw_tests ${agcMetalDir}/tests/D6DrawTests.mm)
target_include_directories(agc_metal_draw_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/core/shader/recompiler ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR} ${ANYPS5_SPIRV_HEADERS_INCLUDE_DIR})
target_link_libraries(agc_metal_draw_tests PRIVATE libSceAgcDriver SDL2-static libc "-framework Metal" "-framework Foundation")
set_source_files_properties(${agcMetalDir}/tests/D6DrawTests.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
set_target_properties(agc_metal_draw_tests PROPERTIES CXX_EXTENSIONS OFF OBJCXX_STANDARD 20)
if(CMAKE_CXX_COMPILER_LAUNCHER)
    set_property(TARGET agc_metal_draw_tests PROPERTY OBJCXX_COMPILER_LAUNCHER ${CMAKE_CXX_COMPILER_LAUNCHER})
endif()
foreach(agcMetalDrawMode IN ITEMS targets primitives)
    add_test(NAME agc_metal_draw_${agcMetalDrawMode} COMMAND agc_metal_draw_tests ${agcMetalDrawMode})
    set_tests_properties(agc_metal_draw_${agcMetalDrawMode} PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1" SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 30 LABELS gpu)
endforeach()
# Headless presentation (no window opens: the PresentationWindow has no Metal layer).
add_test(NAME agc_metal_offscreen_frames COMMAND agc_metal_draw_tests offscreen)
set_tests_properties(agc_metal_offscreen_frames PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1;APS5_METAL_PRESENT_READBACK=1;APS5_METALFX=0" SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 30 LABELS gpu)
# D61: upscaling presents go through the MetalFX spatial scaler (the default).
add_test(NAME agc_metal_metalfx_present COMMAND agc_metal_draw_tests metalfx)
set_tests_properties(agc_metal_metalfx_present PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1;APS5_METAL_PRESENT_READBACK=1" SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 30 LABELS gpu)
# D66: display buffers decoded on the GPU in the present (a 1:1 target, the bilinear scaler).
add_test(NAME agc_metal_present_gpu_decode COMMAND agc_metal_draw_tests decode)
set_tests_properties(agc_metal_present_gpu_decode PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1;APS5_METAL_PRESENT_READBACK=1;APS5_METALFX=0;APS5_PRESENT_SIZE=300x260" SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 30 LABELS gpu)
