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
        -DBUILD_TESTING=OFF
        -DBUILD_THUNKS=OFF
        -DBUILD_FEXCONFIG=OFF
        -DBUILD_FEX_LINUX_TESTS=OFF
        -DENABLE_FEX_ALLOCATOR=OFF
        -DENABLE_ASSERTIONS=OFF
        -DENABLE_CCACHE=OFF
        -DENABLE_LTO=OFF
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target FEXCore FEXCore_Base JemallocLibs --parallel ${APS5_FEX_JOBS}
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ${APS5_FEX_LIBRARIES})

add_executable(aps5-fex src/Main.cpp src/Bridge.cpp src/GuestCpu.cpp src/GuestImage.cpp src/GuestProgram.cpp src/Trampolines.S)
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
        add_test(NAME fex_bridge_tables COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_bridge_tables.py
            ${CMAKE_CURRENT_SOURCE_DIR}/src/Bridge.cpp ${CMAKE_SOURCE_DIR}/core/libs)
    endif()
endif()
