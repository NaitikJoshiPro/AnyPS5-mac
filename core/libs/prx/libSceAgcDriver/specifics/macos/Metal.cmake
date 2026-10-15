# macOS: the AGC driver runs on Metal (docs/mac/M2_DESIGN.md, decision D8). Upstream sources that
# call Vulkan are replaced by Objective-C++ twins; each work package lists its pairs in
# specifics/macos/twins/<package>.cmake as "original=twin" (paths relative to this library), so
# packages never edit the same list. Twin headers live in specifics/macos/overlay, which every
# target of this directory searches before the shared include root, so the PM4 core, the twins and
# the GPU tests see one definition of each class.
set(agcMetalDir ${CMAKE_CURRENT_SOURCE_DIR}/specifics/macos)
set(agcMetalTwins)
file(GLOB agcMetalTwinLists CONFIGURE_DEPENDS ${agcMetalDir}/twins/*.cmake)
list(SORT agcMetalTwinLists)
foreach(agcMetalTwinList IN LISTS agcMetalTwinLists)
    include(${agcMetalTwinList})
endforeach()

set(agcMetalTwinSources)
foreach(agcMetalTwin IN LISTS agcMetalTwins)
    string(REPLACE "=" ";" agcMetalPair "${agcMetalTwin}")
    list(GET agcMetalPair 1 agcMetalTwinSource)
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${agcMetalTwinSource})
        message(FATAL_ERROR "Metal twin ${agcMetalTwinSource} does not exist")
    endif()
    list(APPEND agcMetalTwinSources ${CMAKE_CURRENT_SOURCE_DIR}/${agcMetalTwinSource})
endforeach()
set_source_files_properties(${agcMetalTwinSources} PROPERTIES COMPILE_OPTIONS "-fobjc-arc")

# Metal-only tests, one registration file per work package (tests/<package>.cmake).
if(BUILD_TESTING)
    file(GLOB agcMetalTestLists CONFIGURE_DEPENDS ${agcMetalDir}/tests/*.cmake)
    list(SORT agcMetalTestLists)
    foreach(agcMetalTestList IN LISTS agcMetalTestLists)
        include(${agcMetalTestList})
    endforeach()
endif()

get_property(agcMetalTargets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
set(agcMetalRemoved)
foreach(agcMetalTarget IN LISTS agcMetalTargets)
    get_target_property(agcMetalType ${agcMetalTarget} TYPE)
    if(agcMetalType STREQUAL "UTILITY" OR agcMetalType STREQUAL "INTERFACE_LIBRARY")
        continue()
    endif()
    target_include_directories(${agcMetalTarget} BEFORE PRIVATE ${agcMetalDir}/overlay)
    get_target_property(agcMetalSources ${agcMetalTarget} SOURCES)
    # An original may have several twins (one per work package); all of them replace it.
    set(agcMetalAdded)
    foreach(agcMetalTwin IN LISTS agcMetalTwins)
        string(REPLACE "=" ";" agcMetalPair "${agcMetalTwin}")
        list(GET agcMetalPair 0 agcMetalOriginal)
        list(GET agcMetalPair 1 agcMetalTwinSource)
        foreach(agcMetalCandidate IN ITEMS ${CMAKE_CURRENT_SOURCE_DIR}/${agcMetalOriginal} ${agcMetalOriginal})
            if(${agcMetalCandidate} IN_LIST agcMetalSources OR ${agcMetalCandidate} IN_LIST agcMetalRemoved)
                list(APPEND agcMetalAdded ${CMAKE_CURRENT_SOURCE_DIR}/${agcMetalTwinSource})
                list(REMOVE_ITEM agcMetalSources ${agcMetalCandidate})
                list(APPEND agcMetalRemoved ${agcMetalCandidate})
            endif()
        endforeach()
    endforeach()
    set(agcMetalRemoved)
    set(agcMetalReplaced FALSE)
    if(agcMetalAdded)
        list(REMOVE_DUPLICATES agcMetalAdded)
        list(APPEND agcMetalSources ${agcMetalAdded})
        set(agcMetalReplaced TRUE)
    endif()
    if(agcMetalReplaced)
        set_property(TARGET ${agcMetalTarget} PROPERTY SOURCES ${agcMetalSources})
        set_property(TARGET ${agcMetalTarget} PROPERTY OBJCXX_STANDARD 20)
        if(CMAKE_CXX_COMPILER_LAUNCHER)
            set_property(TARGET ${agcMetalTarget} PROPERTY OBJCXX_COMPILER_LAUNCHER ${CMAKE_CXX_COMPILER_LAUNCHER})
        endif()
        target_link_libraries(${agcMetalTarget} PRIVATE "-framework Metal" "-framework MetalFX" "-framework QuartzCore" "-framework Foundation")
    endif()
endforeach()

# agc_driver_bda_device_tests compiles three driver sources whose twins need the rest of the Metal
# driver (the recorder, the image cache and D4's Metal::Vk image calls): on macOS it links the driver
# library for them instead, as the Metal-only tests do.
if(TARGET agc_driver_bda_device_tests)
    get_target_property(agcBdaSources agc_driver_bda_device_tests SOURCES)
    foreach(agcBdaDriverSource IN ITEMS Resources.mm PlaceholderImages.mm BufferAddress.mm)
        list(REMOVE_ITEM agcBdaSources ${agcMetalDir}/src/${agcBdaDriverSource})
    endforeach()
    set_property(TARGET agc_driver_bda_device_tests PROPERTY SOURCES ${agcBdaSources})
    target_link_libraries(agc_driver_bda_device_tests PRIVATE libSceAgcDriver)
endif()

# libSceAudioIn/libSceAudioOut/libSceVideoOut take SDL from the driver on macOS (one SDL per process, see
# libSceAudioOut/CMakeLists.txt). The Vulkan driver pulls SDL's core in through SDL_Init; the Metal driver does
# not call SDL yet (presentation is D6), so keep SDL_Init (and through it the audio/video subsystems) linked.
if(TARGET libSceAgcDriver)
    target_link_options(libSceAgcDriver PRIVATE "LINKER:-u,_SDL_Init")
endif()
