enable_language(ASM)
include(ExternalProject)
include(ProcessorCount)
ProcessorCount(APS5_FEX_JOBS)
if(APS5_FEX_JOBS EQUAL 0)
    set(APS5_FEX_JOBS 1)
endif()

set(APS5_FEX_SOURCE_DIR ${CMAKE_SOURCE_DIR}/3rdparty/FEX)
set(APS5_FEX_BINARY_DIR ${CMAKE_BINARY_DIR}/3rdparty/FEX)
set(APS5_FEX_LIBRARIES
    ${APS5_FEX_BINARY_DIR}/FEXCore/Source/libFEXCore.a
    ${APS5_FEX_BINARY_DIR}/FEXCore/Source/libFEXCore_Base.a
    ${APS5_FEX_BINARY_DIR}/FEXCore/Source/libJemallocLibs.a
    ${APS5_FEX_BINARY_DIR}/External/fmt/libfmt.a
    ${APS5_FEX_BINARY_DIR}/External/xxhash/cmake_unofficial/libxxhash.a
    ${APS5_FEX_BINARY_DIR}/External/cephes/libcephes_128bit.a
    ${APS5_FEX_BINARY_DIR}/External/SoftFloat-3e/libsoftfloat_3e.a)

ExternalProject_Add(fexcore
    SOURCE_DIR ${APS5_FEX_SOURCE_DIR}
    BINARY_DIR ${APS5_FEX_BINARY_DIR}
    CMAKE_ARGS
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_OSX_ARCHITECTURES=arm64
        -DCMAKE_BUILD_TYPE=Release
        -DTUNE_CPU=none
        -DCMAKE_DISABLE_FIND_PACKAGE_fmt=TRUE
        -DCMAKE_DISABLE_FIND_PACKAGE_xxhash=TRUE
        -DBUILD_TESTING=OFF
        -DBUILD_THUNKS=OFF
        -DBUILD_FEXCONFIG=OFF
        -DBUILD_FEX_LINUX_TESTS=OFF
        -DENABLE_FEX_ALLOCATOR=OFF
        -DENABLE_ASSERTIONS=OFF
        -DENABLE_CCACHE=OFF
        -DENABLE_LTO=OFF
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target FEXCore FEXCore_Base JemallocLibs xxhash --parallel ${APS5_FEX_JOBS}
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ${APS5_FEX_LIBRARIES})

add_executable(aps5-fex src/Main.cpp src/Bridge.cpp src/GuestCpu.cpp src/GuestImage.cpp src/GuestProgram.cpp src/Options.cpp src/Trampolines.S)
add_dependencies(aps5-fex fexcore)
target_compile_features(aps5-fex PRIVATE cxx_std_20)
target_include_directories(aps5-fex PRIVATE
    ${APS5_FEX_SOURCE_DIR}/FEXCore/include
    ${APS5_FEX_BINARY_DIR}/include
    ${APS5_FEX_SOURCE_DIR}/FEXHeaderUtils
    ${APS5_FEX_SOURCE_DIR}/External/fmt/include
    ${APS5_FEX_SOURCE_DIR}/External/range-v3/include
    ${APS5_FEX_SOURCE_DIR}/External/unordered_dense/include)
target_link_libraries(aps5-fex PRIVATE ${APS5_FEX_LIBRARIES})

if(BUILD_TESTING)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME fex_fixtures COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_fex_fixtures.py
            $<TARGET_FILE:relinker> $<TARGET_FILE:aps5-fex> ${CMAKE_BINARY_DIR}/core/libs/libs)
        add_test(NAME fex_licence COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/mac/check_fex_licence.py
            ${CMAKE_SOURCE_DIR})
        add_test(NAME fex_licence_checker COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/tools/tests/test_check_fex_licence.py)
        add_test(NAME fex_bench_smoke COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_fex_bench.py
            $<TARGET_FILE:relinker> $<TARGET_FILE:aps5-fex> ${CMAKE_BINARY_DIR}/core/libs/libs --smoke)
        set_tests_properties(fex_bench_smoke PROPERTIES TIMEOUT 30 SKIP_RETURN_CODE 77)
    endif()
endif()

option(APS5_FEX_BRIDGE_SIGNATURES "Compile the bridge signature table derived from the prx library sources into the runner" OFF)
find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND AND CMAKE_EXPORT_COMPILE_COMMANDS)
    set(APS5_FEX_SIGNATURES_DIR ${CMAKE_CURRENT_BINARY_DIR}/signatures)
    set(APS5_FEX_SIGNATURES_JSON ${APS5_FEX_SIGNATURES_DIR}/BridgeSignatures.json)
    set(APS5_FEX_SIGNATURES_TOOL ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_bridge_signatures.py)
    set(APS5_FEX_PRX_TARGETS libc)
    file(GLOB APS5_FEX_PRX_DIRECTORIES LIST_DIRECTORIES true ${CMAKE_SOURCE_DIR}/core/libs/prx/*)
    foreach(prxDirectory IN LISTS APS5_FEX_PRX_DIRECTORIES)
        get_filename_component(prxName ${prxDirectory} NAME)
        if(TARGET ${prxName})
            list(APPEND APS5_FEX_PRX_TARGETS ${prxName})
        endif()
    endforeach()
    list(REMOVE_DUPLICATES APS5_FEX_PRX_TARGETS)
    add_custom_command(OUTPUT ${APS5_FEX_SIGNATURES_DIR}/BridgeSignatures.inc ${APS5_FEX_SIGNATURES_JSON}
        COMMAND ${Python3_EXECUTABLE} ${APS5_FEX_SIGNATURES_TOOL} generate --compile-commands ${CMAKE_BINARY_DIR}/compile_commands.json
            --root ${CMAKE_SOURCE_DIR} --cache ${APS5_FEX_SIGNATURES_DIR}/cache --json ${APS5_FEX_SIGNATURES_JSON}
            --inc ${APS5_FEX_SIGNATURES_DIR}/BridgeSignatures.inc
        DEPENDS ${APS5_FEX_SIGNATURES_TOOL} ${CMAKE_BINARY_DIR}/compile_commands.json ${APS5_FEX_PRX_TARGETS}
        COMMENT "Deriving the bridge signatures from the prx library sources"
        VERBATIM)
    add_custom_target(aps5-fex-bridge-signatures DEPENDS ${APS5_FEX_SIGNATURES_DIR}/BridgeSignatures.inc ${APS5_FEX_SIGNATURES_JSON})
    if(APS5_FEX_BRIDGE_SIGNATURES)
        set(APS5_FEX_BRIDGE_TARGET aps5-fex)
        if(TARGET aps5-fex-runtime)
            set(APS5_FEX_BRIDGE_TARGET aps5-fex-runtime)
        endif()
        target_sources(${APS5_FEX_BRIDGE_TARGET} PRIVATE src/BridgeSignatures.cpp)
        target_compile_definitions(${APS5_FEX_BRIDGE_TARGET} PRIVATE APS5_FEX_BRIDGE_SIGNATURES)
        target_include_directories(${APS5_FEX_BRIDGE_TARGET} PRIVATE ${APS5_FEX_SIGNATURES_DIR})
        add_dependencies(${APS5_FEX_BRIDGE_TARGET} aps5-fex-bridge-signatures)
    endif()
    if(BUILD_TESTING)
        add_test_executable(fex_bridge_signatures_tests tests/BridgeSignatures.cpp src/BridgeSignatures.cpp)
        target_compile_features(fex_bridge_signatures_tests PRIVATE cxx_std_20)
        target_compile_definitions(fex_bridge_signatures_tests PRIVATE APS5_FEX_BRIDGE_SIGNATURES)
        target_include_directories(fex_bridge_signatures_tests PRIVATE src ${APS5_FEX_SIGNATURES_DIR})
        add_dependencies(fex_bridge_signatures_tests aps5-fex-bridge-signatures)
        add_test(NAME fex_bridge_signatures_table COMMAND fex_bridge_signatures_tests)
        add_test(NAME fex_bridge_signatures COMMAND ${Python3_EXECUTABLE} ${APS5_FEX_SIGNATURES_TOOL} check
            --json ${APS5_FEX_SIGNATURES_JSON} --bridge ${CMAKE_CURRENT_SOURCE_DIR}/src/Bridge.cpp)
        add_test(NAME fex_bridge_signatures_abi COMMAND ${Python3_EXECUTABLE} ${APS5_FEX_SIGNATURES_TOOL} verify
            --json ${APS5_FEX_SIGNATURES_JSON} --work ${APS5_FEX_SIGNATURES_DIR}/abi)
        add_test(NAME fex_bridge_signatures_units COMMAND ${Python3_EXECUTABLE} ${APS5_FEX_SIGNATURES_TOOL} verify
            --json ${APS5_FEX_SIGNATURES_JSON} --work ${APS5_FEX_SIGNATURES_DIR}/units
            --compile-commands ${CMAKE_BINARY_DIR}/compile_commands.json --root ${CMAKE_SOURCE_DIR})
        set_tests_properties(fex_bridge_signatures_units PROPERTIES TIMEOUT 900 LABELS slow)
        add_test(NAME fex_bridge_signatures_generator COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gen_bridge_signatures.py)
        set_tests_properties(fex_bridge_signatures_generator PROPERTIES TIMEOUT 300)
    endif()
elseif(APS5_FEX_BRIDGE_SIGNATURES)
    message(FATAL_ERROR "APS5_FEX_BRIDGE_SIGNATURES needs Python 3 and CMAKE_EXPORT_COMPILE_COMMANDS")
endif()
