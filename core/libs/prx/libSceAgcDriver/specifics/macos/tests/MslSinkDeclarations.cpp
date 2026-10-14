// agc_metal_msl_sink_declarations (DECISIONS D54): MslSinkDeclarations, the MetalBackend pass that
// declares an SSA local at its definition instead of zero-initialized at function scope, checked
// on the CPU (no Metal device). Every case states which locals may move and which must stay: a
// local moves only when its first occurrence is its own definition statement and every other
// occurrence is in that statement's block or a block nested in it. The emitted libraries are
// compiled by agc_metal_msl_corpus and run by the GPU groups.

#include "MetalBackend/MslEmitterState.hpp"
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct Case {
    const char* name;
    std::string locals;
    std::string prologue;
    std::string body;
    std::size_t sunk;
    std::string expectedLocals;
    std::string expectedBody;
};

void Check(const Case& test) {
    std::string locals = test.locals;
    std::string body = test.body;
    const auto sunk = ShaderRecompiler::MslSinkDeclarations(locals, test.prologue, body);
    Require(sunk == test.sunk, std::string(test.name) + ": sank " + std::to_string(sunk) + " local(s), expected " + std::to_string(test.sunk));
    Require(locals == test.expectedLocals, std::string(test.name) + ": locals are\n" + locals + "expected\n" + test.expectedLocals);
    Require(body == test.expectedBody, std::string(test.name) + ": body is\n" + body + "expected\n" + test.expectedBody);
}

}

int main() {
    try {
        const std::string keep = "    uint aps5_guard = 0u;\n";
        const Case cases[] = {
            {"straight line", keep + "    uint v1 = uint(0);\n    float v2 = float(0);\n", "",
                "    v1 = aps5_push[0u];\n    v2 = as_type<float>(v1);\n    aps5_out = v2;\n", 2u,
                keep, "    uint v1 = aps5_push[0u];\n    float v2 = as_type<float>(v1);\n    aps5_out = v2;\n"},
            {"use in a nested block", "    uint v1 = uint(0);\n", "",
                "    v1 = 3u;\n    if (c) {\n        x = v1;\n    }\n", 1u,
                "", "    uint v1 = 3u;\n    if (c) {\n        x = v1;\n    }\n"},
            {"defined in a block, used after it", "    uint v1 = uint(0);\n", "",
                "    if (c) {\n        v1 = 3u;\n    }\n    x = v1;\n", 0u,
                "    uint v1 = uint(0);\n", "    if (c) {\n        v1 = 3u;\n    }\n    x = v1;\n"},
            {"assigned in both arms (phi)", "    uint v1 = uint(0);\n", "",
                "    if (c) {\n        v1 = 3u;\n    } else {\n        v1 = 4u;\n    }\n    x = v1;\n", 0u,
                "    uint v1 = uint(0);\n", "    if (c) {\n        v1 = 3u;\n    } else {\n        v1 = 4u;\n    }\n    x = v1;\n"},
            {"loop-carried (phi before the loop)", "    uint v1 = uint(0);\n", "",
                "    v1 = 0u;\n    for (;;) {\n        v1 = v1 + 1u;\n        if (v1 > 4u) break;\n    }\n", 1u,
                "", "    uint v1 = 0u;\n    for (;;) {\n        v1 = v1 + 1u;\n        if (v1 > 4u) break;\n    }\n"},
            {"read before its definition", "    uint v1 = uint(0);\n", "",
                "    x = v1;\n    v1 = 2u;\n", 0u,
                "    uint v1 = uint(0);\n", "    x = v1;\n    v1 = 2u;\n"},
            {"self-referencing definition", "    uint v1 = uint(0);\n", "",
                "    v1 = v1 + 1u;\n", 0u,
                "    uint v1 = uint(0);\n", "    v1 = v1 + 1u;\n"},
            {"used by the prologue", "    uint v1 = uint(0);\n", "    aps5_lds[0] = v1;\n",
                "    v1 = 2u;\n", 0u,
                "    uint v1 = uint(0);\n", "    v1 = 2u;\n"},
            {"definition after a brace-less if", "    uint v1 = uint(0);\n", "",
                "    if (c)\n        v1 = 2u;\n    x = v1;\n", 0u,
                "    uint v1 = uint(0);\n", "    if (c)\n        v1 = 2u;\n    x = v1;\n"},
            {"definition in a switch body", "    uint v1 = uint(0);\n", "",
                "    switch (m) {\n    case 0u:\n        v1 = 2u;\n        x = v1;\n        break;\n    default:\n        break;\n    }\n", 0u,
                "    uint v1 = uint(0);\n", "    switch (m) {\n    case 0u:\n        v1 = 2u;\n        x = v1;\n        break;\n    default:\n        break;\n    }\n"},
            {"preprocessor line", "    uint v1 = uint(0);\n", "",
                "    v1 = 2u;\n#if APS5_X\n    x = v1;\n#endif\n", 0u,
                "    uint v1 = uint(0);\n", "    v1 = 2u;\n#if APS5_X\n    x = v1;\n#endif\n"},
            {"second lane half and longer names", "    uint v12 = uint(0);\n    uint v12_h = uint(0);\n    uint v123 = uint(0);\n", "",
                "    v12 = 1u;\n    v12_h = 2u;\n    v123 = v12 + v12_h;\n", 3u,
                "", "    uint v12 = 1u;\n    uint v12_h = 2u;\n    uint v123 = v12 + v12_h;\n"},
            {"not a zero-initialized SSA local", "    uint v1 = 5u;\n    uint aps5_t0 = uint(0);\n    uint v2[4] = {};\n", "",
                "    v1 = 2u;\n    aps5_t0 = 1u;\n", 0u,
                "    uint v1 = 5u;\n    uint aps5_t0 = uint(0);\n    uint v2[4] = {};\n", "    v1 = 2u;\n    aps5_t0 = 1u;\n"},
            {"identifier inside a comment", "    uint v1 = uint(0);\n", "",
                "    // v1 is read below\n    v1 = 2u;\n    x = v1;\n", 1u,
                "", "    // v1 is read below\n    uint v1 = 2u;\n    x = v1;\n"},
        };
        for (const auto& test : cases) Check(test);
        std::printf("msl declaration sinking: %zu cases passed\n", std::size(cases));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
