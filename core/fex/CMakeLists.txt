enable_language(ASM)
include(ExternalProject)
include(ProcessorCount)
ProcessorCount(APS5_FEX_JOBS)
if(APS5_FEX_JOBS EQUAL 0)
    set(APS5_FEX_JOBS 1)
endif()
# The outer ninja runs alongside; keep FEX's own build from taking every core.
if(APS5_FEX_JOBS GREATER 4)
    set(APS5_FEX_JOBS 4)
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
        # GDBJIT.cpp includes GDB's GPL-3.0 <gdb/jit-reader.h> when this is on; FEX turns it on by default
        # whenever the header is installed. tools/mac/check_fex_licence.py checks this line.
        -DENABLE_GDB_SYMBOLS=OFF
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target FEXCore FEXCore_Base JemallocLibs xxhash --parallel ${APS5_FEX_JOBS}
    INSTALL_COMMAND ""
    # FEX is vendored and patched in tree: rebuild it whenever the runner builds (a no-op ninja when unchanged).
    BUILD_ALWAYS TRUE
    BUILD_BYPRODUCTS ${APS5_FEX_LIBRARIES})

add_library(aps5-fex-runtime STATIC src/Bridge.cpp src/BridgeSignatures.cpp src/ConsoleSyscalls.cpp src/GuestCall.cpp src/GuestGuards.cpp src/GuestCpu.cpp src/GuestImage.cpp src/GuestProgram.cpp src/GuestSampler.cpp src/GuestStats.cpp src/Options.cpp src/Trampolines.S)
target_sources(aps5-fex-runtime PRIVATE src/AotSeeds.cpp src/CodeCache.cpp src/CodePool.cpp)
add_dependencies(aps5-fex-runtime fexcore)
# Header-only: the displaced guest window (GuestDisplaced.hpp) the fault handler must leave to libc.
target_include_directories(aps5-fex-runtime PRIVATE ${CMAKE_SOURCE_DIR}/core/libs)
target_compile_features(aps5-fex-runtime PUBLIC cxx_std_20)
target_include_directories(aps5-fex-runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${APS5_FEX_SOURCE_DIR}/FEXCore/include
    ${APS5_FEX_BINARY_DIR}/include
    ${APS5_FEX_SOURCE_DIR}/FEXHeaderUtils
    ${APS5_FEX_SOURCE_DIR}/External/fmt/include
    ${APS5_FEX_SOURCE_DIR}/External/range-v3/include
    ${APS5_FEX_SOURCE_DIR}/External/unordered_dense/include
    ${APS5_FEX_SOURCE_DIR}/External/xxhash)
target_link_libraries(aps5-fex-runtime PUBLIC ${APS5_FEX_LIBRARIES})

add_executable(aps5-fex src/Main.cpp)
target_link_libraries(aps5-fex PRIVATE aps5-fex-runtime)

if(BUILD_TESTING)
    add_test_executable(fex_code_pool_tests tests/CodePool.cpp src/CodePool.cpp)
    target_compile_features(fex_code_pool_tests PRIVATE cxx_std_20)
    target_include_directories(fex_code_pool_tests PRIVATE src ${APS5_FEX_SOURCE_DIR}/FEXCore/include ${APS5_FEX_BINARY_DIR}/include
        ${APS5_FEX_SOURCE_DIR}/External/fmt/include)
    add_dependencies(fex_code_pool_tests fexcore)
    add_test(NAME fex_code_pool COMMAND fex_code_pool_tests)
    add_test_executable(fex_aot_seeds_tests tests/AotSeeds.cpp src/AotSeeds.cpp)
    target_compile_features(fex_aot_seeds_tests PRIVATE cxx_std_20)
    target_include_directories(fex_aot_seeds_tests PRIVATE src)
    add_test(NAME fex_aot_seeds COMMAND fex_aot_seeds_tests)
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
        add_test(NAME fex_code_churn COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_fex_fixtures.py
            $<TARGET_FILE:relinker> $<TARGET_FILE:aps5-fex> ${CMAKE_BINARY_DIR}/core/libs/libs --only code-churn)
        set_tests_properties(fex_code_churn PROPERTIES TIMEOUT 300 ENVIRONMENT APS5_FEX_CLEAR_CODE=1)
        add_test(NAME fex_displaced_churn COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_fex_fixtures.py
            $<TARGET_FILE:relinker> $<TARGET_FILE:aps5-fex> ${CMAKE_BINARY_DIR}/core/libs/libs --only displaced-churn)
        set_tests_properties(fex_displaced_churn PROPERTIES TIMEOUT 300 ENVIRONMENT APS5_FEX_CLEAR_CODE=1)
        add_test(NAME fex_disk_cache COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_fex_cache.py
            $<TARGET_FILE:relinker> $<TARGET_FILE:aps5-fex> ${CMAKE_BINARY_DIR}/core/libs/libs)
        set_tests_properties(fex_disk_cache PROPERTIES TIMEOUT 600 SKIP_RETURN_CODE 77)
        add_test(NAME fex_coverage_tool COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/tests/test_fex_coverage.py)

        add_test_executable(fex_x87_convert_tests tests/X87Convert.cpp)
        target_compile_features(fex_x87_convert_tests PRIVATE cxx_std_20)
        target_include_directories(fex_x87_convert_tests PRIVATE src)
        add_test(NAME fex_x87_convert COMMAND fex_x87_convert_tests)

        add_test_executable(aps5-fex-asm tests/asm/AsmHarness.cpp)
        target_link_libraries(aps5-fex-asm PRIVATE aps5-fex-runtime)
        set(APS5_FEX_ASM_ROOT ${APS5_FEX_SOURCE_DIR}/unittests/ASM)
        set(APS5_FEX_ASM_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/asm)
        add_test(NAME fex_asm_prepare COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/asm/prepare_asm_tests.py
            ${APS5_FEX_ASM_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/asm ${APS5_FEX_ASM_OUTPUT})
        set_tests_properties(fex_asm_prepare PROPERTIES TIMEOUT 1200 FIXTURES_SETUP fex_asm_prepared)
        add_test(NAME fex_asm_prepare_tests COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_prepare_asm_tests.py)
        set(APS5_FEX_ASM_NASM "" CACHE FILEPATH "nasm for the FEX ASM tests (nasm 2.16 gives the bytes FEX recorded its results with); empty uses nasm on PATH")
        if(APS5_FEX_ASM_NASM)
            set_property(TEST fex_asm_prepare fex_asm_prepare_tests APPEND PROPERTY ENVIRONMENT NASM=${APS5_FEX_ASM_NASM})
        endif()
        file(GLOB_RECURSE APS5_FEX_ASM_TESTS CONFIGURE_DEPENDS RELATIVE ${APS5_FEX_ASM_ROOT} ${APS5_FEX_ASM_ROOT}/*.asm)
        foreach(asmTest IN LISTS APS5_FEX_ASM_TESTS)
            set(asmOptions)
            if(asmTest MATCHES "^SelfModifyingCode/")
                set(asmOptions --option FEX_SMCCHECKS=full)
            endif()
            add_test(NAME fex_asm/${asmTest} COMMAND aps5-fex-asm --variant default --option FEX_MAXINST=500 --option FEX_MULTIBLOCK=1
                ${asmOptions} ${APS5_FEX_ASM_OUTPUT}/${asmTest}.cfg)
            set_tests_properties(fex_asm/${asmTest} PROPERTIES TIMEOUT 30 SKIP_RETURN_CODE 125 FIXTURES_REQUIRED fex_asm_prepared
                LABELS fex_asm)
            add_test(NAME fex_asm_full/${asmTest} COMMAND aps5-fex-asm --variant full --option FEX_MAXINST=1 --option FEX_MULTIBLOCK=0
                ${asmOptions} ${APS5_FEX_ASM_OUTPUT}/${asmTest}.cfg)
            set_tests_properties(fex_asm_full/${asmTest} PROPERTIES TIMEOUT 30 SKIP_RETURN_CODE 125 FIXTURES_REQUIRED fex_asm_prepared
                LABELS "fex_asm_full;slow")
        endforeach()
    endif()
endif()

# A test-only PS5 library with one export per ABI class the bridge marshals (tests/prx/libAps5BridgeAbi),
# NID-patched like the libraries in core/libs. The abi_*.cpp fixtures load it from a staged libs directory.
if(BUILD_TESTING AND TARGET nid_patcher AND TARGET libc)
    set(APS5_FEX_ABI_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/abi-libs)
    add_library(libAps5BridgeAbi SHARED tests/prx/libAps5BridgeAbi/Export.cpp)
    set_target_properties(libAps5BridgeAbi PROPERTIES PREFIX "" SUFFIX .prx
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/abi-unpatched)
    add_custom_command(OUTPUT ${APS5_FEX_ABI_LIBRARY_DIR}/libAps5BridgeAbi.prx
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:libAps5BridgeAbi> ${APS5_FEX_ABI_LIBRARY_DIR}/libAps5BridgeAbi.prx
        COMMAND codesign --remove-signature ${APS5_FEX_ABI_LIBRARY_DIR}/libAps5BridgeAbi.prx
        COMMAND $<TARGET_FILE:nid_patcher> libAps5BridgeAbi --preserve-exports $<TARGET_FILE:libc>
            ${APS5_FEX_ABI_LIBRARY_DIR}/libAps5BridgeAbi.prx
        COMMAND codesign --force --sign - ${APS5_FEX_ABI_LIBRARY_DIR}/libAps5BridgeAbi.prx
        DEPENDS nid_patcher libAps5BridgeAbi libc
        COMMENT "Patching NIDs for libAps5BridgeAbi")
    add_custom_target(aps5-fex-abi-library ALL DEPENDS ${APS5_FEX_ABI_LIBRARY_DIR}/libAps5BridgeAbi.prx)
endif()

# Without the table the bridge maps every export positionally, which is wrong for the exports listed in
# docs/mac/M1_DESIGN.md (package D): long double math, structs, stack arguments, hidden results.
option(APS5_FEX_BRIDGE_SIGNATURES "Compile the bridge signature table derived from the prx library sources into the runner" ON)
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
    if(TARGET libAps5BridgeAbi)
        list(APPEND APS5_FEX_PRX_TARGETS libAps5BridgeAbi)
    endif()
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
