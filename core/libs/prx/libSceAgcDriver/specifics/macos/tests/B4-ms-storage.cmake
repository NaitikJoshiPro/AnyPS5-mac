# Work package B4 (MSIMG): emulated multisampled storage images (MslAbi Version 6). Included by
# Metal.cmake when BUILD_TESTING is on.
# agc_metal_msl_ms_storage: MSL shape on the CPU, then a Metal compile of every program (no dispatch).
add_test_executable(agc_metal_msl_ms_storage_tests ${agcMetalDir}/tests/MslMsStorage.mm)
target_include_directories(agc_metal_msl_ms_storage_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/core/shader/recompiler $<TARGET_PROPERTY:shader_recompiler_agc_driver,INTERFACE_INCLUDE_DIRECTORIES> ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR} ${ANYPS5_SPIRV_HEADERS_INCLUDE_DIR})
target_link_libraries(agc_metal_msl_ms_storage_tests PRIVATE libSceAgcDriver SDL2-static libc "-framework Metal" "-framework Foundation")
set_source_files_properties(${agcMetalDir}/tests/MslMsStorage.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
set_target_properties(agc_metal_msl_ms_storage_tests PROPERTIES CXX_EXTENSIONS OFF OBJCXX_STANDARD 20)
if(CMAKE_CXX_COMPILER_LAUNCHER)
    set_property(TARGET agc_metal_msl_ms_storage_tests PROPERTY OBJCXX_COMPILER_LAUNCHER ${CMAKE_CXX_COMPILER_LAUNCHER})
endif()
add_test(NAME agc_metal_msl_ms_storage COMMAND agc_metal_msl_ms_storage_tests)
set_tests_properties(agc_metal_msl_ms_storage PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1" SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 60)
