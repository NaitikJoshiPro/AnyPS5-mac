# UPSYNC3 (upstream 721db55dc) test twins: Metal ports of the tests/Recorder.cpp modes upstream added since 0214069e3,
# and the upstream execution tests that build their own Vulkan instance. The image table tests (tests/execution/
# ImageTable*.cpp) run unchanged on Metal (MslAbi Version 15 image and sampler tables).
list(APPEND agcMetalTwins
        # --read-list-only, --space-pending-writes-only, --pending-write-snapshot-only, --queued-write-groups-only,
        # --pending-key-stores-only, --shadowed-key-stores-only, --kept-unit-settle-only.
        "tests/Recorder.cpp=specifics/macos/tests/Upsync3RecorderModesA.mm"
        # --sparse-run-import-only, --sparse-run-copied-only, --draw-snapshot-epoch-only, --slice-dirty-only,
        # --metadata-pass-only, --resident-present-only (tests/ResidentPresent.cpp), --tiled-resolve-only,
        # --pipeline-cache-file-only (over the Metal pipeline archive, as D5RecorderModes.mm maps --pipeline-cache-only).
        "tests/Recorder.cpp=specifics/macos/tests/Upsync3RecorderModesB.mm"
        # Second round (upstream 5d9dcc42d2): --import-watch-only (the decision checks; the probe refuses on Metal,
        # D19) and --sample-dumps-only (without the Vulkan CoverPass half).
        "tests/Recorder.cpp=specifics/macos/tests/Upsync3RecorderModesC.mm"
        # Third round (upstream 0d078f75d3): --device-selection-only (MetalDevice.mm SelectDevice's rule) and
        # --sampled-storage-base-level-usage-only; tests/HostImportLifetime.cpp's cases over Metal imports.
        "tests/Recorder.cpp=specifics/macos/tests/Upsync3RecorderModesD.mm"
        "tests/HostImportLifetime.cpp=specifics/macos/tests/HostImportLifetimeTwin.mm"
        # Builds its own Vulkan instance upstream; the twin runs it unchanged over the Metal image context.
        "tests/execution/PartiallyResidentSnapshot.cpp=specifics/macos/tests/PartiallyResidentSnapshotTwin.mm"
        # Second round: builds its own Vulkan instance upstream; Run runs unchanged over the Metal image context.
        "tests/execution/DepthSurfaceLastWriter.cpp=specifics/macos/tests/DepthSurfaceLastWriterTwin.mm"
        # Second round: native multisampled depth sampling has no Metal path yet; every mode skips naming it.
        "tests/execution/NativeMultisampleDepth.cpp=specifics/macos/tests/NativeMultisampleDepthTwin.mm"
        # The GPU presentation pass is Vulkan-only (metal-present-pass): the twin runs the letterbox checks and skips.
        "tests/PresentationPass.cpp=specifics/macos/tests/PresentationPassTwin.mm"
        # Third round: upstream execution tests of Vulkan-only behaviour (DCC key presync and resource cache growth of
        # stage-A dispatch preparation, the pipeline journal) or of Metal driver features not written yet
        # (render pass totals, GPU indirect draws): PendingTwins.mm skips each with its reason.
        "tests/execution/DccKeyPresync.cpp=specifics/macos/tests/PendingTwins.mm"
        "tests/execution/ResourceCacheGrowth.cpp=specifics/macos/tests/PendingTwins.mm"
        "tests/execution/PipelineJournal.cpp=specifics/macos/tests/PendingTwins.mm"
        "tests/execution/AddressPasses.cpp=specifics/macos/tests/PendingTwins.mm"
        "tests/execution/RectListIndirect.cpp=specifics/macos/tests/PendingTwins.mm"
        # Third round: Vulkan fault-injection tests (command batch submit/fence failures, descriptor layout/pool failures).
        "tests/CommandBatch.cpp=specifics/macos/tests/PendingTwins.mm"
        "tests/DescriptorCacheFailure.cpp=specifics/macos/tests/PendingTwins.mm"
        # Third round: execution tests that run on Metal through a thin twin that includes the upstream source:
        # the f16 op check reads the MSL text (no SPIR-V for Metal targets); guest memory lifetime has Metal host
        # imports (no libvulkan probe); the mesh tests see MeshShadingEXT (NGG runs as MSL mesh functions, D23);
        # pipelined dispatch skips its APS5_PIPELINE_DISPATCH item run (metal-dispatch-recipes).
        "tests/execution/Float16Native.cpp=specifics/macos/tests/Float16NativeTwin.mm"
        "tests/execution/GuestMemoryLifetime.cpp=specifics/macos/tests/GuestMemoryLifetimeTwin.mm"
        "tests/execution/MeshDrawRelocation.cpp=specifics/macos/tests/MeshDrawRelocationTwin.mm"
        "tests/execution/MeshProvokingVertex.cpp=specifics/macos/tests/MeshProvokingVertexTwin.mm"
        "tests/execution/MeshIndexCache.cpp=specifics/macos/tests/MeshIndexCacheTwin.mm"
        "tests/execution/PipelinedDispatch.cpp=specifics/macos/tests/PipelinedDispatchTwin.mm"
        # Third round: upstream counts the vector window's slot variables in result.spirv; the twin counts them in
        # the MSL source (MslAluMath.cpp window narrowing).
        "tests/execution/VectorWindowNarrowing.cpp=specifics/macos/tests/VectorWindowNarrowingTwin.mm"
        # Third round: upstream checks tessellation spacing and factor bounds in the SPIR-V of the hull and domain; the
        # twin checks the linked software tessellator's partitioning (LinkMslTessellation, D24/D26) and the MSL source.
        "tests/execution/Tessellation.cpp=specifics/macos/tests/TessellationTwin.mm"
        # Third round: --lock-budget counts the Workgroup-storage bytes of result.spirv upstream; the twin counts the
        # MSL source's threadgroup arrays (the other modes run upstream's main unchanged).
        "tests/execution/LdsMixedWidthAtomics.cpp=specifics/macos/tests/LdsMixedWidthAtomicsTwin.mm"
        # Stencil replay is the Vulkan fallback for devices without stencil export; Metal exports [[stencil]] natively, so
        # the twin keeps the capability, drops the replay-only bind count and refusals, and runs every stencil result check.
        "tests/execution/StencilReplay.cpp=specifics/macos/tests/StencilReplayTwin.mm"
        "tests/execution/StencilNullFeedback.cpp=specifics/macos/tests/StencilNullFeedbackTwin.mm"
)
