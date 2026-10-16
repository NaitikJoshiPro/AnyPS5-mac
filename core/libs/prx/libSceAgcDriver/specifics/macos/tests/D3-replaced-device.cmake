# A replacing device commits on the queue the host imports' residency set is attached to (MetalDevice.mm
# SharedQueue): Metal-only GPU test. Included by Metal.cmake when BUILD_TESTING is on.
add_test_executable(agc_metal_replaced_device_queue_tests ${agcMetalDir}/tests/ReplacedDeviceQueue.mm)
target_include_directories(agc_metal_replaced_device_queue_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/core/shader/recompiler ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR} ${ANYPS5_SPIRV_HEADERS_INCLUDE_DIR})
target_link_libraries(agc_metal_replaced_device_queue_tests PRIVATE libSceAgcDriver SDL2-static libc "-framework Metal" "-framework Foundation")
set_source_files_properties(${agcMetalDir}/tests/ReplacedDeviceQueue.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
set_target_properties(agc_metal_replaced_device_queue_tests PROPERTIES CXX_EXTENSIONS OFF OBJCXX_STANDARD 20)
if(CMAKE_CXX_COMPILER_LAUNCHER)
    set_property(TARGET agc_metal_replaced_device_queue_tests PROPERTY OBJCXX_COMPILER_LAUNCHER ${CMAKE_CXX_COMPILER_LAUNCHER})
endif()
add_test(NAME agc_metal_replaced_device_queue COMMAND agc_metal_replaced_device_queue_tests)
set_tests_properties(agc_metal_replaced_device_queue PROPERTIES SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 30 LABELS gpu)
