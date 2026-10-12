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
