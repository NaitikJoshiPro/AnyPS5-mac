# Work packages D4/D5: Metal-only tests of the image path (src/MetalImages.h). Included by Metal.cmake when BUILD_TESTING is on.
# agc_metal_texture_detile: CPU (de)tiler against ColorTargetLayout and round trips (no Metal device).
# agc_metal_images: uploads, heap views, placeholders and samplers (Metal objects, no GPU work).
add_test_executable(agc_metal_texture_tests ${agcMetalDir}/tests/MetalTextureTests.mm)
target_include_directories(agc_metal_texture_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/core/shader/recompiler $<TARGET_PROPERTY:shader_recompiler_agc_driver,INTERFACE_INCLUDE_DIRECTORIES> ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR} ${ANYPS5_SPIRV_HEADERS_INCLUDE_DIR})
target_link_libraries(agc_metal_texture_tests PRIVATE libSceAgcDriver libc "-framework Metal" "-framework Foundation")
set_source_files_properties(${agcMetalDir}/tests/MetalTextureTests.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
set_target_properties(agc_metal_texture_tests PROPERTIES CXX_EXTENSIONS OFF OBJCXX_STANDARD 20)
if(CMAKE_CXX_COMPILER_LAUNCHER)
    set_property(TARGET agc_metal_texture_tests PROPERTY OBJCXX_COMPILER_LAUNCHER ${CMAKE_CXX_COMPILER_LAUNCHER})
endif()
add_test(NAME agc_metal_texture_detile COMMAND agc_metal_texture_tests --detile)
add_test(NAME agc_metal_images COMMAND agc_metal_texture_tests --images)
add_test(NAME agc_metal_image_mip_overflow COMMAND agc_metal_texture_tests --mip-overflow)
add_test(NAME agc_metal_sampler_reduction COMMAND agc_metal_texture_tests --reduction)
add_test(NAME agc_metal_texture_detile_gpu COMMAND agc_metal_texture_tests --detile-gpu)
add_test(NAME agc_metal_color_transfer COMMAND agc_metal_texture_tests --color-transfer)
set_tests_properties(agc_metal_texture_detile agc_metal_images agc_metal_image_mip_overflow agc_metal_sampler_reduction agc_metal_texture_detile_gpu agc_metal_color_transfer PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
