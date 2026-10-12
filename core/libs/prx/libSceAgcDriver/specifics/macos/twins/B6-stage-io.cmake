# Work package B6 (stage interfaces): upstream tests whose shaders are prebuilt SPIR-V = Metal twins with MSL shaders.
# ProvokingVertexTwin.mm runs tests/execution/ProvokingVertex.cpp's draws and checks (compiled unchanged with its main
# renamed) with MSL vertex/fragment functions instead of the GLSL-compiled SPIR-V ones.
list(APPEND agcMetalTwins
        "tests/execution/ProvokingVertex.cpp=specifics/macos/tests/ProvokingVertexTwin.mm"
)
