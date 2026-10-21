# SKIPFIX5 (UPSYNC5 skip check): upstream 6b6c701636 tests that skipped on Metal over Vulkan feature probes.
list(APPEND agcMetalTwins
        # The Metal device is a depth-range-unrestricted one (MTLViewport takes any znear/zfar, MTLDepthClipModeClamp
        # clamps): the twin runs upstream's unrestricted-path cases; the vertex-shader remap fallback is Vulkan-only.
        "tests/execution/DepthRangeRemapDraw.cpp=specifics/macos/tests/DepthRangeRemapDrawTwin.mm"
        # Rect lists are expanded in the Metal vertex function, not by SPIR-V tessellation stages: the twin draws
        # upstream's rect list with guest programs and runs its corner checks.
        "tests/execution/RectListCorner.cpp=specifics/macos/tests/RectListCornerTwin.mm"
)
# Data-only draw cache hits (upstream DrawDataHitUserWords.cpp) skip without graphics pipeline libraries unless
# APS5_NO_GPL names the library-less draw path, the only one Metal has (metal-graphics-pipeline-libraries): on Metal
# both registrations run that path, as agc_driver_draw_cache_reuse_off does.
foreach(agcMetalNoGplTest IN ITEMS agc_driver_draw_data_hit_user_words agc_driver_draw_data_hit_user_words_verify)
    if(TEST ${agcMetalNoGplTest})
        set_property(TEST ${agcMetalNoGplTest} APPEND PROPERTY ENVIRONMENT "APS5_NO_GPL=1")
    endif()
endforeach()
