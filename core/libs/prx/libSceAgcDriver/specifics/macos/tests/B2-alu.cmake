# Work package B2 (ALU and f64): Metal-only tests. Included by Metal.cmake when BUILD_TESTING is on.
# agc_metal_alu_helpers: the MetalBackend Alu/Float16/Float64 prelude on the GPU against CPU IEEE references.
add_test_executable(agc_metal_alu_helpers_tests ${agcMetalDir}/tests/MslAluHelpers.mm)
target_include_directories(agc_metal_alu_helpers_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/core/shader/recompiler $<TARGET_PROPERTY:shader_recompiler_agc_driver,INTERFACE_INCLUDE_DIRECTORIES> ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR} ${ANYPS5_SPIRV_HEADERS_INCLUDE_DIR})
target_link_libraries(agc_metal_alu_helpers_tests PRIVATE libSceAgcDriver SDL2-static libc "-framework Metal" "-framework Foundation")
set_source_files_properties(${agcMetalDir}/tests/MslAluHelpers.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
set_target_properties(agc_metal_alu_helpers_tests PROPERTIES CXX_EXTENSIONS OFF OBJCXX_STANDARD 20)
if(CMAKE_CXX_COMPILER_LAUNCHER)
    set_property(TARGET agc_metal_alu_helpers_tests PROPERTY OBJCXX_COMPILER_LAUNCHER ${CMAKE_CXX_COMPILER_LAUNCHER})
endif()
add_test(NAME agc_metal_alu_helpers COMMAND agc_metal_alu_helpers_tests)
set_tests_properties(agc_metal_alu_helpers PROPERTIES SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 60)

# agc_metal_native_f32_flush (DECISIONS D54): the native f32 flush translation against the front end's
# emulated flush, bit for bit, over special, random and near-2^-126 operands in six float modes.
add_test_executable(agc_metal_native_f32_flush_tests ${agcMetalDir}/tests/NativeF32Flush.cpp)
target_include_directories(agc_metal_native_f32_flush_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/core/shader/recompiler ${CMAKE_CURRENT_SOURCE_DIR}/tests/execution ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR} ${ANYPS5_SPIRV_HEADERS_INCLUDE_DIR})
target_link_libraries(agc_metal_native_f32_flush_tests PRIVATE libSceAgcDriver SDL2-static libc)
set_target_properties(agc_metal_native_f32_flush_tests PROPERTIES CXX_EXTENSIONS OFF)
add_test(NAME agc_metal_native_f32_flush COMMAND agc_metal_native_f32_flush_tests)
set_tests_properties(agc_metal_native_f32_flush PROPERTIES SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 30)
# agc_metal_nan_test_elimination (DECISIONS D63): NaN tests of values that cannot be NaNs dropped, bit for bit against
# the full emulation (GPU).
add_test_executable(agc_metal_nan_test_elimination_tests ${agcMetalDir}/tests/NanTestElimination.cpp)
target_include_directories(agc_metal_nan_test_elimination_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/core/shader/recompiler ${CMAKE_CURRENT_SOURCE_DIR}/tests/execution ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR} ${ANYPS5_SPIRV_HEADERS_INCLUDE_DIR})
target_link_libraries(agc_metal_nan_test_elimination_tests PRIVATE libSceAgcDriver SDL2-static libc)
set_target_properties(agc_metal_nan_test_elimination_tests PROPERTIES CXX_EXTENSIONS OFF)
add_test(NAME agc_metal_nan_test_elimination COMMAND agc_metal_nan_test_elimination_tests)
set_tests_properties(agc_metal_nan_test_elimination PROPERTIES SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 30)
