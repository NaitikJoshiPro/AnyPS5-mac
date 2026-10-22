# UPSYNC3 round 3 (upstream 0d078f75d3): Metal-only registrations of upstream tests. Included by Metal.cmake when
# BUILD_TESTING is on. agc_driver_draw_cache_relocation skips on Metal (no graphics pipeline libraries,
# metal-graphics-pipeline-libraries); like upstream's agc_driver_draw_cache_reuse_off, the APS5_NO_GPL run checks the
# relocated draw cache over flexible render pipelines instead.
if(TARGET agc_driver_draw_cache_relocation_tests)
    add_test(NAME agc_driver_draw_cache_relocation_off COMMAND agc_driver_draw_cache_relocation_tests)
    set_tests_properties(agc_driver_draw_cache_relocation_off PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1;APS5_NO_GPL=1"
            SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 60 FAIL_REGULAR_EXPRESSION "FATAL:" LABELS gpu)
endif()
# Upstream registers these without SKIP_RETURN_CODE; on Metal they exit 77 with an allowlisted reason: the fault
# injection tests through PendingTwins.mm (twins/UPSYNC3-tests.cmake, metal-vulkan-internal-tests), the pipeline journal,
# pipeline library and VkPipeline store modes through GraphicsTwin.mm (metal-pipeline-journal,
# metal-graphics-pipeline-libraries, metal-vulkan-internal-tests).
foreach(agcMetalSkippedTest IN ITEMS agc_command_batch_failure agc_descriptor_cache_failure agc_pipeline_journal
        agc_pipeline_library_clear_waits_for_build agc_pipeline_store_lazy_buffer_pool)
    if(TEST ${agcMetalSkippedTest})
        set_tests_properties(${agcMetalSkippedTest} PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endforeach()
