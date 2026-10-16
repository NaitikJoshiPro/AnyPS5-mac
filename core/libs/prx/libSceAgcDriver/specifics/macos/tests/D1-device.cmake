# Work package D1 (device, submission, compute): Metal-only tests. Included by Metal.cmake when BUILD_TESTING is on.
add_test_executable(agc_metal_device_tests ${agcMetalDir}/tests/D1DeviceTests.mm)
target_include_directories(agc_metal_device_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/core/shader/recompiler ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR} ${ANYPS5_SPIRV_HEADERS_INCLUDE_DIR})
target_link_libraries(agc_metal_device_tests PRIVATE libSceAgcDriver SDL2-static libc "-framework Metal" "-framework Foundation")
set_source_files_properties(${agcMetalDir}/tests/D1DeviceTests.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
set_target_properties(agc_metal_device_tests PROPERTIES CXX_EXTENSIONS OFF OBJCXX_STANDARD 20)
if(CMAKE_CXX_COMPILER_LAUNCHER)
    set_property(TARGET agc_metal_device_tests PROPERTY OBJCXX_COMPILER_LAUNCHER ${CMAKE_CXX_COMPILER_LAUNCHER})
endif()
foreach(agcMetalDeviceMode IN ITEMS bindings submission profile imports memory residency import_window)
    add_test(NAME agc_metal_device_${agcMetalDeviceMode} COMMAND agc_metal_device_tests ${agcMetalDeviceMode})
    set_tests_properties(agc_metal_device_${agcMetalDeviceMode} PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1" SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 30)
endforeach()
# APS5_WRITE_WATCH_IMPORTS=watch: imports stay in the write watch (DECISIONS D74).
add_test(NAME agc_metal_device_imports_watched COMMAND agc_metal_device_tests imports_watched)
set_tests_properties(agc_metal_device_imports_watched PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1;APS5_WRITE_WATCH_IMPORTS=watch" SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 30)
# D19.5 teardown hygiene: a failed MSL compile, a dispatch in flight and a wrapped static page at teardown
# (through ~VulkanDevice, and with the device still alive at exit()). Run under validation via gpu_suite.
add_test(NAME agc_metal_device_teardown COMMAND agc_metal_device_tests teardown)
set_tests_properties(agc_metal_device_teardown PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1;APS5_METAL_TEARDOWN_LOG=1" SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 30)
add_test(NAME agc_metal_device_teardown_exit COMMAND agc_metal_device_tests teardown_exit)
set_tests_properties(agc_metal_device_teardown_exit PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1;APS5_METAL_TEARDOWN_LOG=1" SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 30
    PASS_REGULAR_EXPRESSION "\\[metal\\] teardown at exit: recorder idle, 0 failed syncs, [1-9][0-9]* wrappers dropped, device residency set removed from the queue")
