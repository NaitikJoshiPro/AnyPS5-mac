# Work package B1 (MetalBackend core): Metal-only tests. Included by Metal.cmake when BUILD_TESTING is on.
add_test_executable(agc_metal_msl_compile_tests ${agcMetalDir}/tests/MslCompileCheck.mm)
target_include_directories(agc_metal_msl_compile_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/core/shader/recompiler $<TARGET_PROPERTY:shader_recompiler_agc_driver,INTERFACE_INCLUDE_DIRECTORIES> ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR} ${ANYPS5_SPIRV_HEADERS_INCLUDE_DIR})
target_link_libraries(agc_metal_msl_compile_tests PRIVATE libSceAgcDriver SDL2-static libc "-framework Metal" "-framework Foundation")
set_source_files_properties(${agcMetalDir}/tests/MslCompileCheck.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
set_target_properties(agc_metal_msl_compile_tests PROPERTIES CXX_EXTENSIONS OFF OBJCXX_STANDARD 20)
if(CMAKE_CXX_COMPILER_LAUNCHER)
    set_property(TARGET agc_metal_msl_compile_tests PROPERTY OBJCXX_COMPILER_LAUNCHER ${CMAKE_CXX_COMPILER_LAUNCHER})
endif()
add_test(NAME agc_metal_msl_compile COMMAND agc_metal_msl_compile_tests)
set_tests_properties(agc_metal_msl_compile PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1" SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 60)

# agc_metal_msl_sink_declarations (DECISIONS D54): MslSinkDeclarations on the CPU (no Metal device).
add_test_executable(agc_metal_msl_sink_declarations_tests ${agcMetalDir}/tests/MslSinkDeclarations.cpp)
target_include_directories(agc_metal_msl_sink_declarations_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/core/shader/recompiler ${CMAKE_SOURCE_DIR}/core/shader/recompiler/MetalBackend/include $<TARGET_PROPERTY:shader_recompiler_agc_driver,INTERFACE_INCLUDE_DIRECTORIES> ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR} ${ANYPS5_SPIRV_HEADERS_INCLUDE_DIR})
target_link_libraries(agc_metal_msl_sink_declarations_tests PRIVATE libSceAgcDriver SDL2-static libc)
add_test(NAME agc_metal_msl_sink_declarations COMMAND agc_metal_msl_sink_declarations_tests)
set_tests_properties(agc_metal_msl_sink_declarations PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 30)
