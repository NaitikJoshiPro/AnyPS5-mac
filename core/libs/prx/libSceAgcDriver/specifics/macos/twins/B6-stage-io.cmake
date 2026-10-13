# Work package B6 (stage interfaces): upstream tests whose shaders are prebuilt SPIR-V = Metal twins with MSL shaders.
# ProvokingVertexTwin.mm runs tests/execution/ProvokingVertex.cpp's draws and checks (compiled unchanged with its main
# renamed) with MSL vertex/fragment functions instead of the GLSL-compiled SPIR-V ones.
list(APPEND agcMetalTwins
        "tests/execution/ProvokingVertex.cpp=specifics/macos/tests/ProvokingVertexTwin.mm"
        # LayeredVolumeDraw.cpp's mesh half inspects SPIR-V; the twin checks and draws the MSL mesh function's layer.
        "tests/execution/LayeredVolumeDraw.cpp=specifics/macos/tests/LayeredVolumeDrawTwin.mm"
        # Mesh render pipelines (stage 5) beside the pipeline cache (MetalMesh.mm).
        "Graphics/src/PipelineCache.cpp=specifics/macos/src/MetalMesh.mm"
)
