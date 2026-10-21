# UPSYNC5 (upstream 6b6c701636) test twins. Included by Metal.cmake after the library's tests are registered.
# Upstream registers these tests/Graphics.cpp modes without SKIP_RETURN_CODE; on Metal GraphicsTwin.mm exits 77 for
# them with the reason it prints: Vulkan-only (sample location grids, VkFence creation, the pipeline journal's
# long and external paths: metal-sample-location-grid, metal-vulkan-internal-tests, metal-pipeline-journal) or a
# Metal twin not written yet (kept-bytes and heap mirror modes: metal-graphics-twin-pending-modes).
foreach(agcMetalSkippedTest IN ITEMS agc_driver_sample_location_grid_fallback agc_command_batch_fence_failure
        agc_pipeline_journal_long_paths agc_pipeline_journal_extended_paths agc_copy_input_kept_bytes
        agc_depth_copy_kept_bytes agc_heap_mirror_unwatch agc_heap_mirror_unwatch_no_space_cache
        agc_heap_mirror_unwatched_import agc_sparse_heap_mirror_unwatch agc_sparse_heap_mirror_unwatch_no_space_cache
        agc_heap_mirror_run_unwatch agc_heap_mirror_run_unwatch_no_space_cache)
    if(TEST ${agcMetalSkippedTest})
        set_tests_properties(${agcMetalSkippedTest} PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endforeach()
