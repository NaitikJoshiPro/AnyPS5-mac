// The fixed-function tessellator of the Metal tessellation path (DECISIONS D24), shared by the MSL
// mesh functions and the CPU (MslTessellator.hpp, which includes this file twice: once as C++ code,
// once stringified into the MSL source). It is a port of the D3D11 reference tessellator
// (CHWTessellator, Copyright (c) Microsoft Corporation, MIT licence; the copy in Mesa's
// src/gallium/auxiliary/tessellator/tessellator.cpp), which RDNA2's tessellator follows: the same
// factor processing, 16.16 fixed-point point placement, ring order and stitching. Instead of filling
// point and index arrays it answers three questions without storage: how many primitives a patch
// has (aps5_tess_primitive_count), which domain points primitive k of a contiguous window uses
// (aps5_tess_primitives) and where domain point i lies (aps5_tess_point), so mesh threadgroups can
// each tessellate a slice of one patch.
//
// The text must stay valid C++17 and MSL: no preprocessor directives, no string or character
// literals, no lambdas or pointers to functions, every float rule NaN-proof through bit tests (Metal
// compiles with fast math). The includer defines:
//   APS5_TESS_SOURCE(...)        the code itself, or its text
//   APS5_TESS_FN                 function specifiers
//   APS5_TESS_TABLE              specifiers of the constant tables
//   APS5_TESS_THREAD             address space of references to the caller's structs
//   APS5_TESS_OUT                address space of the primitive output array
//   APS5_TESS_AS_UINT(f), APS5_TESS_AS_FLOAT(u), APS5_TESS_CEIL(f)
APS5_TESS_SOURCE(

struct Aps5TessFactors {
    float outer[4];
    float inner[2];
};

struct Aps5TessPoint {
    float u;
    float v;
};

struct Aps5TessContext {
    uint invFloor;
    uint invCeil;
    uint halfFraction;
    int halfPoints;
    int splitPoint;
};

struct Aps5TessPatch {
    int domain;
    int partitioning;
    int topology;
    int culled;
    int minimum;
    int outsideParity[4];
    int insideParity[2];
    Aps5TessContext outside[4];
    Aps5TessContext inside[2];
    int outsidePoints[4];
    int insidePoints[2];
    int insideBase;
    int points;
    int lines;
    int linePoints;
    int detailParity;
    int densityParity;
    Aps5TessContext detail;
    Aps5TessContext density;
};

struct Aps5TessSink {
    int next;
    int first;
    int end;
    int clockwise;
    int patched;
    int insideDelta;
    int insideBad;
    int insideReplacement;
    int outsideBase;
    int outsideDelta;
    int outsideBad;
    int outsideReplacement;
    int patched2;
    int invertBase;
    int invertEnd;
    int cornerBad;
    int cornerReplacement;
};

APS5_TESS_TABLE uint aps5_tess_reciprocal[65] = {
    0xffffffffu, 0x10000u, 0x8000u, 0x5555u, 0x4000u, 0x3333u, 0x2aabu, 0x2492u, 0x2000u,
    0x1c72u, 0x199au, 0x1746u, 0x1555u, 0x13b1u, 0x1249u, 0x1111u, 0x1000u,
    0xf0fu, 0xe39u, 0xd79u, 0xccdu, 0xc31u, 0xba3u, 0xb21u, 0xaabu,
    0xa3du, 0x9d9u, 0x97bu, 0x925u, 0x8d4u, 0x889u, 0x842u, 0x800u,
    0x7c2u, 0x788u, 0x750u, 0x71cu, 0x6ebu, 0x6bdu, 0x690u, 0x666u,
    0x63eu, 0x618u, 0x5f4u, 0x5d1u, 0x5b0u, 0x591u, 0x572u, 0x555u,
    0x539u, 0x51fu, 0x505u, 0x4ecu, 0x4d5u, 0x4beu, 0x4a8u, 0x492u,
    0x47eu, 0x46au, 0x457u, 0x444u, 0x432u, 0x421u, 0x410u, 0x400u};

APS5_TESS_TABLE int aps5_tess_final_position[33] = {0, 32, 16, 8, 17, 4, 18, 9, 19, 2, 20, 10, 21, 5, 22, 11, 23, 1, 24, 12, 25, 6, 26, 13, 27, 3, 28, 14, 29, 7, 30, 15, 31};
APS5_TESS_TABLE int aps5_tess_loop_start[33] = {1, 1, 17, 9, 9, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
APS5_TESS_TABLE int aps5_tess_loop_end[33] = {0, 0, 17, 17, 25, 25, 25, 25, 29, 29, 29, 29, 29, 29, 29, 29, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 32};

APS5_TESS_FN bool aps5_tess_nan(float value) {
    const uint bits = APS5_TESS_AS_UINT(value);
    return (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u;
}

APS5_TESS_FN bool aps5_tess_positive(float value) {
    const uint bits = APS5_TESS_AS_UINT(value);
    return (bits & 0x80000000u) == 0u && bits != 0u && !aps5_tess_nan(value);
}

APS5_TESS_FN float aps5_tess_flush(float value) {
    const uint bits = APS5_TESS_AS_UINT(value);
    if ((bits & 0x7fffffffu) < 0x00800000u) return APS5_TESS_AS_FLOAT(bits & 0x80000000u);
    return value;
}

APS5_TESS_FN float aps5_tess_fmin(float a, float b) {
    const float fa = aps5_tess_flush(a);
    const float fb = aps5_tess_flush(b);
    if (aps5_tess_nan(fb)) return a;
    if ((APS5_TESS_AS_UINT(fa) & 0x7fffffffu) == 0u && (APS5_TESS_AS_UINT(fb) & 0x7fffffffu) == 0u) return (APS5_TESS_AS_UINT(fa) & 0x80000000u) != 0u ? a : b;
    return fa < fb ? a : b;
}

APS5_TESS_FN float aps5_tess_fmax(float a, float b) {
    const float fa = aps5_tess_flush(a);
    const float fb = aps5_tess_flush(b);
    if (aps5_tess_nan(fb)) return a;
    if ((APS5_TESS_AS_UINT(fa) & 0x7fffffffu) == 0u && (APS5_TESS_AS_UINT(fb) & 0x7fffffffu) == 0u) return (APS5_TESS_AS_UINT(fb) & 0x80000000u) != 0u ? a : b;
    return fa >= fb ? a : b;
}

APS5_TESS_FN float aps5_tess_clamp(float value, float lower, float upper) {
    return aps5_tess_fmin(upper, aps5_tess_fmax(lower, value));
}

APS5_TESS_FN bool aps5_tess_above(float value, float bound) {
    return !aps5_tess_nan(value) && value > bound;
}

APS5_TESS_FN uint aps5_tess_to_fixed(float value) {
    const uint bits = APS5_TESS_AS_UINT(value);
    const int exponent = int((bits & 0x7f800000u) >> 23u) - 127;
    if (exponent == 128 && (bits & 0x007fffffu) != 0u) return 0u;
    if ((bits & 0x80000000u) == 0u && bits >= 0x47000000u) return 0xffffffffu;
    if ((bits & 0x80000000u) != 0u) return 0u;
    if (exponent < -17) return 0u;
    uint mantissa = (bits & 0x007fffffu) | 0x00800000u;
    const int extra = 7 - exponent;
    if (extra >= 0) {
        const uint lsb = 1u << uint(extra);
        const uint halfway = lsb >> 1u;
        if ((mantissa & lsb) != 0u || (mantissa & (lsb - 1u)) > halfway) mantissa += halfway;
        return mantissa >> uint(extra);
    }
    return mantissa << uint(-extra);
}

APS5_TESS_FN float aps5_tess_to_float(uint value) {
    return float(value >> 16u) + float(value & 0xffffu) * (1.0f / 65536.0f);
}

APS5_TESS_FN bool aps5_tess_even(float value) {
    return (int(value) & 1) == 0;
}

APS5_TESS_FN uint aps5_tess_ceil_fixed(uint value) {
    return (value & 0xffffu) != 0u ? (value & 0x7fff0000u) + 0x10000u : value;
}

APS5_TESS_FN int aps5_tess_remove_msb(int value) {
    int check = value <= 0x0000ffff ? (value <= 0x000000ff ? 0x00000080 : 0x00008000) : (value <= 0x00ffffff ? 0x00800000 : int(0x80000000u));
    for (int i = 0; i < 8; ++i) {
        if ((value & check) != 0) return value & ~check;
        check = int(uint(check) >> 1u);
    }
    return 0;
}

APS5_TESS_FN int aps5_tess_point_count(uint factor, int odd) {
    if (odd != 0) return int((aps5_tess_ceil_fixed(0x8000u + (factor + 1u) / 2u) * 2u) >> 16u);
    return int((aps5_tess_ceil_fixed((factor + 1u) / 2u) * 2u) >> 16u) + 1;
}

APS5_TESS_FN Aps5TessContext aps5_tess_context(uint factor, int odd) {
    Aps5TessContext context;
    uint halfway = (factor + 1u) / 2u;
    if (odd != 0 || halfway == 0x8000u) halfway += 0x8000u;
    const uint lower = halfway & 0x7fff0000u;
    const uint upper = aps5_tess_ceil_fixed(halfway);
    context.halfFraction = halfway - lower;
    context.halfPoints = int(upper >> 16u);
    if (upper == lower) {
        context.splitPoint = context.halfPoints + 1;
    } else if (odd != 0) {
        context.splitPoint = lower == 0x10000u ? 0 : (aps5_tess_remove_msb(int(lower >> 16u) - 1) << 1) + 1;
    } else {
        context.splitPoint = (aps5_tess_remove_msb(int(lower >> 16u)) << 1) + 1;
    }
    int lowerSegments = int((lower * 2u) >> 16u);
    int upperSegments = int((upper * 2u) >> 16u);
    if (odd != 0) {
        lowerSegments -= 1;
        upperSegments -= 1;
    }
    context.invFloor = aps5_tess_reciprocal[lowerSegments];
    context.invCeil = aps5_tess_reciprocal[upperSegments];
    return context;
}

APS5_TESS_FN uint aps5_tess_place(Aps5TessContext context, int point, int odd) {
    bool flip = false;
    if (point >= context.halfPoints) {
        point = (context.halfPoints << 1) - point;
        if (odd != 0) point -= 1;
        flip = true;
    }
    if (point == context.halfPoints) return 0x8000u;
    const uint upperIndex = uint(point);
    uint lowerIndex = upperIndex;
    if (point > context.splitPoint) lowerIndex -= 1u;
    const uint lowerLocation = lowerIndex * context.invFloor;
    const uint upperLocation = upperIndex * context.invCeil;
    uint location = lowerLocation * (0x10000u - context.halfFraction) + upperLocation * context.halfFraction;
    location = (location + 0x8000u) >> 16u;
    return flip ? 0x10000u - location : location;
}

APS5_TESS_FN Aps5TessPatch aps5_tess_process(int domain, int partitioning, int topology, Aps5TessFactors factors) {
    Aps5TessPatch patch;
    patch.domain = domain;
    patch.partitioning = partitioning;
    patch.topology = topology;
    patch.culled = 0;
    patch.minimum = 0;
    patch.points = 0;
    patch.lines = 0;
    patch.linePoints = 0;
    patch.insideBase = 0;
    patch.detailParity = 0;
    patch.densityParity = 0;
    for (int i = 0; i < 4; ++i) {
        patch.outsideParity[i] = 0;
        patch.outsidePoints[i] = 0;
        patch.outside[i] = aps5_tess_context(0x10000u, 1);
    }
    for (int i = 0; i < 2; ++i) {
        patch.insideParity[i] = 0;
        patch.insidePoints[i] = 0;
        patch.inside[i] = aps5_tess_context(0x10000u, 1);
    }
    patch.detail = patch.inside[0];
    patch.density = patch.inside[0];
    const bool integer = partitioning == 0 || partitioning == 1;
    const int parity = partitioning == 2 ? 1 : 0;
    float lower = 1.0f;
    float upper = 64.0f;
    if (partitioning == 3) lower = 2.0f;
    if (partitioning == 2) upper = 63.0f;
    if (domain == 0) {
        if (!aps5_tess_positive(factors.outer[0]) || !aps5_tess_positive(factors.outer[1])) {
            patch.culled = 1;
            return patch;
        }
        const float density = APS5_TESS_CEIL(aps5_tess_clamp(factors.outer[0], 1.0f, 64.0f));
        float detail = aps5_tess_clamp(factors.outer[1], lower, upper);
        if (integer) {
            detail = APS5_TESS_CEIL(detail);
            patch.detailParity = aps5_tess_even(detail) ? 0 : 1;
        } else {
            patch.detailParity = parity;
        }
        const uint fixedDetail = aps5_tess_to_fixed(detail);
        patch.detail = aps5_tess_context(fixedDetail, patch.detailParity);
        patch.linePoints = aps5_tess_point_count(fixedDetail, patch.detailParity);
        patch.densityParity = aps5_tess_even(density) ? 0 : 1;
        const uint fixedDensity = aps5_tess_to_fixed(density);
        patch.density = aps5_tess_context(fixedDensity, patch.densityParity);
        patch.lines = aps5_tess_point_count(fixedDensity, patch.densityParity) - 1;
        patch.points = patch.linePoints * patch.lines;
        return patch;
    }
    const int edges = domain == 1 ? 3 : 4;
    const int axes = domain == 1 ? 1 : 2;
    float outer[4];
    float inner[2];
    for (int edge = 0; edge < edges; ++edge) {
        if (!aps5_tess_positive(factors.outer[edge])) {
            patch.culled = 1;
            return patch;
        }
    }
    for (int edge = 0; edge < edges; ++edge) {
        outer[edge] = aps5_tess_clamp(factors.outer[edge], lower, upper);
        if (integer) outer[edge] = APS5_TESS_CEIL(outer[edge]);
    }
    if (partitioning == 2) {
        const float bound = 1.0f + 0.0000152587890625f / 2.0f;
        bool frame = false;
        for (int edge = 0; edge < edges; ++edge) frame = frame || outer[edge] > bound;
        if (domain == 2) frame = frame || aps5_tess_above(factors.inner[0], bound) || aps5_tess_above(factors.inner[1], bound);
        if (frame) lower = 1.0f + 0.0000152587890625f;
    }
    for (int axis = 0; axis < axes; ++axis) {
        inner[axis] = aps5_tess_clamp(factors.inner[axis], lower, upper);
        if (integer) inner[axis] = APS5_TESS_CEIL(inner[axis]);
    }
    uint outerFixed[4];
    uint innerFixed[2];
    for (int edge = 0; edge < edges; ++edge) {
        patch.outsideParity[edge] = integer ? (aps5_tess_even(outer[edge]) ? 0 : 1) : parity;
        outerFixed[edge] = aps5_tess_to_fixed(outer[edge]);
    }
    for (int axis = 0; axis < axes; ++axis) {
        patch.insideParity[axis] = integer ? ((aps5_tess_even(inner[axis]) || inner[axis] == 1.0f) ? 0 : 1) : parity;
        innerFixed[axis] = aps5_tess_to_fixed(inner[axis]);
    }
    if (integer || parity != 0) {
        bool ones = true;
        for (int edge = 0; edge < edges; ++edge) ones = ones && outerFixed[edge] == 0x10000u;
        for (int axis = 0; axis < axes; ++axis) ones = ones && innerFixed[axis] == 0x10000u;
        if (ones) {
            patch.minimum = 1;
            patch.points = edges;
            return patch;
        }
    }
    for (int edge = 0; edge < edges; ++edge) {
        patch.outside[edge] = aps5_tess_context(outerFixed[edge], patch.outsideParity[edge]);
        patch.outsidePoints[edge] = aps5_tess_point_count(outerFixed[edge], patch.outsideParity[edge]);
        patch.points += patch.outsidePoints[edge];
    }
    patch.points -= edges;
    for (int axis = 0; axis < axes; ++axis) {
        patch.inside[axis] = aps5_tess_context(innerFixed[axis], patch.insideParity[axis]);
        const int count = aps5_tess_point_count(innerFixed[axis], patch.insideParity[axis]);
        const int least = patch.insideParity[axis] != 0 ? 4 : 3;
        patch.insidePoints[axis] = count > least ? count : least;
    }
    patch.insideBase = patch.points;
    if (domain == 1) {
        const int rings = (patch.insidePoints[0] >> 1) - 1;
        if (patch.insideParity[0] != 0) patch.points += 3 * (rings * (rings + 1) - rings);
        else patch.points += 3 * (rings * (rings + 1)) + 1;
    } else {
        patch.points += (patch.insidePoints[0] - 2) * (patch.insidePoints[1] - 2);
    }
    return patch;
}

APS5_TESS_FN Aps5TessPoint aps5_tess_fixed_point(uint u, uint v) {
    Aps5TessPoint point;
    point.u = aps5_tess_to_float(u);
    point.v = aps5_tess_to_float(v);
    return point;
}

APS5_TESS_FN Aps5TessPoint aps5_tess_point(APS5_TESS_THREAD const Aps5TessPatch& patch, int index) {
    if (patch.domain == 0) {
        const int line = index / patch.linePoints;
        const int point = index - line * patch.linePoints;
        return aps5_tess_fixed_point(aps5_tess_place(patch.detail, point, patch.detailParity), aps5_tess_place(patch.density, line, patch.densityParity));
    }
    if (patch.minimum != 0) {
        if (patch.domain == 1) {
            if (index == 0) return aps5_tess_fixed_point(0u, 0x10000u);
            if (index == 1) return aps5_tess_fixed_point(0u, 0u);
            return aps5_tess_fixed_point(0x10000u, 0u);
        }
        if (index == 0) return aps5_tess_fixed_point(0u, 0u);
        if (index == 1) return aps5_tess_fixed_point(0x10000u, 0u);
        if (index == 2) return aps5_tess_fixed_point(0x10000u, 0x10000u);
        return aps5_tess_fixed_point(0u, 0x10000u);
    }
    int offset = 0;
    if (patch.domain == 1) {
        for (int edge = 0; edge < 3; ++edge) {
            const int end = patch.outsidePoints[edge] - 1;
            if (index < offset + end) {
                const int p = index - offset;
                const int q = (edge & 1) != 0 ? p : end - p;
                const uint parameter = aps5_tess_place(patch.outside[edge], q, patch.outsideParity[edge]);
                if (edge == 0) return aps5_tess_fixed_point(0u, parameter);
                return aps5_tess_fixed_point(parameter, edge == 2 ? 0x10000u - parameter : 0u);
            }
            offset += end;
        }
        const int odd = patch.insideParity[0];
        const int rings = patch.insidePoints[0] >> 1;
        for (int ring = 1; ring < rings; ++ring) {
            const int start = ring;
            const int end = patch.insidePoints[0] - 1 - start;
            const int count = end - start;
            if (index >= offset + 3 * count) {
                offset += 3 * count;
                continue;
            }
            const int edge = (index - offset) / count;
            const int p = start + (index - offset - edge * count);
            uint perpendicular = aps5_tess_place(patch.inside[0], start, odd) * 0xaaaau;
            perpendicular = (perpendicular + 0x8000u) >> 16u;
            const int q = (edge & 1) != 0 ? p : end - (p - start);
            const uint parameter = aps5_tess_place(patch.inside[0], q, odd);
            const uint along = parameter - (perpendicular + 1u) / 2u;
            if (edge == 0) return aps5_tess_fixed_point(perpendicular, along);
            if (edge == 1) return aps5_tess_fixed_point(along, perpendicular);
            return aps5_tess_fixed_point(along, 0x10000u - along - perpendicular);
        }
        return aps5_tess_fixed_point(0x5555u, 0x5555u);
    }
    for (int edge = 0; edge < 4; ++edge) {
        const int end = patch.outsidePoints[edge] - 1;
        if (index < offset + end) {
            const int p = index - offset;
            const int q = (edge == 1 || edge == 2) ? p : end - p;
            const uint parameter = aps5_tess_place(patch.outside[edge], q, patch.outsideParity[edge]);
            if ((edge & 1) != 0) return aps5_tess_fixed_point(parameter, edge == 3 ? 0x10000u : 0u);
            return aps5_tess_fixed_point(edge == 2 ? 0x10000u : 0u, parameter);
        }
        offset += end;
    }
    const int least = patch.insidePoints[0] < patch.insidePoints[1] ? patch.insidePoints[0] : patch.insidePoints[1];
    const int rings = least >> 1;
    for (int ring = 1; ring < rings; ++ring) {
        const int start = ring;
        const int ends[2] = {patch.insidePoints[0] - 1 - start, patch.insidePoints[1] - 1 - start};
        for (int edge = 0; edge < 4; ++edge) {
            const int along = (edge + 1) & 1;
            const int across = edge & 1;
            const int count = ends[along] - start;
            if (index >= offset + count) {
                offset += count;
                continue;
            }
            const int perpendicularPoint = edge < 2 ? start : ends[across];
            const uint perpendicular = aps5_tess_place(patch.inside[across], perpendicularPoint, patch.insideParity[across]);
            const int p = start + (index - offset);
            const int q = (edge == 1 || edge == 2) ? p : ends[along] - (p - start);
            const uint parameter = aps5_tess_place(patch.inside[along], q, patch.insideParity[along]);
            if (along != 0) return aps5_tess_fixed_point(perpendicular, parameter);
            return aps5_tess_fixed_point(parameter, perpendicular);
        }
    }
    const int p = index - offset;
    if (patch.insidePoints[0] > patch.insidePoints[1] && patch.insideParity[1] == 0) {
        return aps5_tess_fixed_point(aps5_tess_place(patch.inside[0], rings + p, patch.insideParity[0]), 0x8000u);
    }
    const int end = patch.insidePoints[1] - 1 - rings;
    return aps5_tess_fixed_point(0x8000u, aps5_tess_place(patch.inside[1], end - p, patch.insideParity[1]));
}

APS5_TESS_FN int aps5_tess_patch_index(APS5_TESS_THREAD const Aps5TessSink& sink, int index) {
    if (sink.patched != 0) {
        if (index >= sink.outsideBase) return index == sink.outsideBad ? sink.outsideReplacement : index + sink.outsideDelta;
        return index == sink.insideBad ? sink.insideReplacement : index + sink.insideDelta;
    }
    if (sink.patched2 != 0) {
        if (index >= sink.invertBase) return index == sink.cornerBad ? sink.cornerReplacement : sink.invertEnd - index;
        if (index == sink.cornerBad) return sink.cornerReplacement;
    }
    return index;
}

APS5_TESS_FN void aps5_tess_triangle(APS5_TESS_THREAD Aps5TessSink& sink, APS5_TESS_OUT int* out, int a, int b, int c) {
    if (sink.next >= sink.first && sink.next < sink.end) {
        const int at = (sink.next - sink.first) * 3;
        out[at] = aps5_tess_patch_index(sink, a);
        out[at + 1] = aps5_tess_patch_index(sink, sink.clockwise != 0 ? b : c);
        out[at + 2] = aps5_tess_patch_index(sink, sink.clockwise != 0 ? c : b);
    }
    sink.next += 1;
}

APS5_TESS_FN bool aps5_tess_skip(APS5_TESS_THREAD Aps5TessSink& sink, int triangles) {
    if (sink.next + triangles <= sink.first || sink.next >= sink.end) {
        sink.next += triangles;
        return true;
    }
    return false;
}

APS5_TESS_FN void aps5_tess_stitch_regular(APS5_TESS_THREAD Aps5TessSink& sink, APS5_TESS_OUT int* out, bool trapezoid, int diagonals, int insidePoints, int inside, int outside) {
    if (trapezoid) {
        aps5_tess_triangle(sink, out, outside, outside + 1, inside);
        outside += 1;
    }
    int p = 0;
    if (diagonals == 0) {
        for (p = 0; p < insidePoints - 1; ++p) {
            aps5_tess_triangle(sink, out, inside, outside, outside + 1);
            aps5_tess_triangle(sink, out, inside, outside + 1, inside + 1);
            inside += 1;
            outside += 1;
        }
    } else if (diagonals == 1) {
        for (p = 0; p < insidePoints / 2 - 1; ++p) {
            aps5_tess_triangle(sink, out, outside, outside + 1, inside);
            aps5_tess_triangle(sink, out, inside, outside + 1, inside + 1);
            inside += 1;
            outside += 1;
        }
        aps5_tess_triangle(sink, out, outside, inside + 1, inside);
        aps5_tess_triangle(sink, out, outside, outside + 1, inside + 1);
        inside += 1;
        outside += 1;
        p += 2;
        for (; p < insidePoints; ++p) {
            aps5_tess_triangle(sink, out, outside, outside + 1, inside);
            aps5_tess_triangle(sink, out, inside, outside + 1, inside + 1);
            inside += 1;
            outside += 1;
        }
    } else {
        for (p = 0; p < insidePoints / 2; ++p) {
            aps5_tess_triangle(sink, out, outside, inside + 1, inside);
            aps5_tess_triangle(sink, out, outside, outside + 1, inside + 1);
            inside += 1;
            outside += 1;
        }
        for (; p < insidePoints - 1; ++p) {
            aps5_tess_triangle(sink, out, inside, outside, outside + 1);
            aps5_tess_triangle(sink, out, inside, outside + 1, inside + 1);
            inside += 1;
            outside += 1;
        }
    }
    if (trapezoid) aps5_tess_triangle(sink, out, outside, outside + 1, inside);
}

APS5_TESS_FN void aps5_tess_stitch_transition(APS5_TESS_THREAD Aps5TessSink& sink, APS5_TESS_OUT int* out, int inside, int insideHalf, int insideParity, int outside, int outsideHalf, int outsideParity) {
    if (insideParity != 0) insideHalf -= 1;
    if (outsideParity != 0) outsideHalf -= 1;
    const int startA = aps5_tess_loop_start[insideHalf];
    const int startB = aps5_tess_loop_start[outsideHalf];
    const int endA = aps5_tess_loop_end[insideHalf];
    const int endB = aps5_tess_loop_end[outsideHalf];
    const int start = startA < startB ? startA : startB;
    const int end = endA > endB ? endA : endB;
    if (aps5_tess_final_position[0] < outsideHalf) {
        aps5_tess_triangle(sink, out, outside, outside + 1, inside);
        outside += 1;
    }
    for (int i = start; i <= end; ++i) {
        if (aps5_tess_final_position[i] < insideHalf) {
            aps5_tess_triangle(sink, out, inside, outside, inside + 1);
            inside += 1;
        }
        if (aps5_tess_final_position[i] < outsideHalf) {
            aps5_tess_triangle(sink, out, outside, outside + 1, inside);
            outside += 1;
        }
    }
    if (insideParity != outsideParity || insideParity != 0) {
        if (insideParity == outsideParity) {
            aps5_tess_triangle(sink, out, inside, outside, inside + 1);
            aps5_tess_triangle(sink, out, inside + 1, outside, outside + 1);
            inside += 1;
            outside += 1;
        } else if (insideParity == 0) {
            aps5_tess_triangle(sink, out, inside, outside, outside + 1);
            outside += 1;
        } else {
            aps5_tess_triangle(sink, out, inside, outside, inside + 1);
            inside += 1;
        }
    }
    for (int i = end; i >= start; --i) {
        if (aps5_tess_final_position[i] < outsideHalf) {
            aps5_tess_triangle(sink, out, outside, outside + 1, inside);
            outside += 1;
        }
        if (aps5_tess_final_position[i] < insideHalf) {
            aps5_tess_triangle(sink, out, inside, outside, inside + 1);
            inside += 1;
        }
    }
    if (aps5_tess_final_position[0] < outsideHalf) {
        aps5_tess_triangle(sink, out, outside, outside + 1, inside);
        outside += 1;
    }
}

APS5_TESS_FN void aps5_tess_stitch(APS5_TESS_THREAD Aps5TessSink& sink, APS5_TESS_OUT int* out, int triangles, bool transition, int insidePoints, int inside, int insideHalf, int insideParity, int outside, int outsideHalf, int outsideParity) {
    if (aps5_tess_skip(sink, triangles)) return;
    const int before = sink.next;
    if (transition) aps5_tess_stitch_transition(sink, out, inside, insideHalf, insideParity, outside, outsideHalf, outsideParity);
    else aps5_tess_stitch_regular(sink, out, true, 2, insidePoints, inside, outside);
    sink.next = before + triangles;
}

APS5_TESS_FN void aps5_tess_set_patch(APS5_TESS_THREAD Aps5TessSink& sink, int insideDelta, int insideBad, int insideReplacement, int outsideCount, int outsideOffset, int outsideReplacement) {
    sink.patched = 1;
    sink.insideDelta = insideDelta;
    sink.insideBad = insideBad;
    sink.insideReplacement = insideReplacement;
    sink.outsideBase = insideBad + 1;
    sink.outsideDelta = outsideOffset - sink.outsideBase;
    sink.outsideBad = sink.outsideBase + outsideCount - 1;
    sink.outsideReplacement = outsideReplacement;
}

APS5_TESS_FN void aps5_tess_triangle_rings(APS5_TESS_THREAD const Aps5TessPatch& patch, APS5_TESS_THREAD Aps5TessSink& sink, APS5_TESS_OUT int* out) {
    const int rings = (patch.insidePoints[0] + 1) >> 1;
    int outsideHalf[3];
    int outsideParity[3];
    int outsidePoints[3];
    for (int edge = 0; edge < 3; ++edge) {
        outsideHalf[edge] = patch.outside[edge].halfPoints;
        outsideParity[edge] = patch.outsideParity[edge];
        outsidePoints[edge] = patch.outsidePoints[edge];
    }
    int insideOffset = patch.insideBase;
    int outsideOffset = 0;
    for (int ring = 1; ring < rings; ++ring) {
        const int insidePoints = patch.insidePoints[0] - 2 * ring;
        const int firstInside = insideOffset;
        const int firstOutside = outsideOffset;
        for (int edge = 0; edge < 3; ++edge) {
            const int triangles = insidePoints + outsidePoints[edge] - 2;
            int insideBase = insideOffset;
            int outsideBase = outsideOffset;
            if (edge == 2) {
                aps5_tess_set_patch(sink, insideOffset, insidePoints - 1, firstInside, outsidePoints[edge], outsideOffset, firstOutside);
                insideBase = 0;
                outsideBase = sink.outsideBase;
            }
            aps5_tess_stitch(sink, out, triangles, ring == 1, insidePoints, insideBase, patch.inside[0].halfPoints, patch.insideParity[0], outsideBase, outsideHalf[edge], outsideParity[edge]);
            sink.patched = 0;
            outsideOffset += outsidePoints[edge] - 1;
            insideOffset += insidePoints - 1;
            outsidePoints[edge] = insidePoints;
        }
        if (ring == 1) {
            for (int edge = 0; edge < 3; ++edge) {
                outsideHalf[edge] = patch.inside[0].halfPoints;
                outsideParity[edge] = patch.insideParity[0];
            }
        }
    }
    if (patch.insideParity[0] != 0) aps5_tess_triangle(sink, out, outsideOffset, outsideOffset + 1, outsideOffset + 2);
}

APS5_TESS_FN void aps5_tess_quad_rings(APS5_TESS_THREAD const Aps5TessPatch& patch, APS5_TESS_THREAD Aps5TessSink& sink, APS5_TESS_OUT int* out) {
    const int rows[2] = {(patch.insidePoints[0] + 1) >> 1, (patch.insidePoints[1] + 1) >> 1};
    const int rings = rows[0] < rows[1] ? rows[0] : rows[1];
    const int degenerate[2] = {patch.insideParity[1] == 0 ? rows[1] - 1 : -1, patch.insideParity[0] == 0 ? rows[0] - 1 : -1};
    int outsideHalf[4];
    int outsideParity[4];
    int outsidePoints[4];
    for (int edge = 0; edge < 4; ++edge) {
        outsideHalf[edge] = patch.outside[edge].halfPoints;
        outsideParity[edge] = patch.outsideParity[edge];
        outsidePoints[edge] = patch.outsidePoints[edge];
    }
    int insideOffset = patch.insideBase;
    int outsideOffset = 0;
    for (int ring = 1; ring < rings; ++ring) {
        const int insidePoints[2] = {patch.insidePoints[0] - 2 * ring, patch.insidePoints[1] - 2 * ring};
        const int firstInside = insideOffset;
        const int firstOutside = outsideOffset;
        for (int edge = 0; edge < 4; ++edge) {
            const int axis = (edge + 1) & 1;
            const int triangles = insidePoints[axis] + outsidePoints[edge] - 2;
            int insideBase = insideOffset;
            int outsideBase = outsideOffset;
            if (edge == 3) {
                if (ring == degenerate[axis]) {
                    sink.patched2 = 1;
                    sink.invertBase = insideOffset + 1;
                    sink.cornerBad = outsideOffset + outsidePoints[edge] - 1;
                    sink.cornerReplacement = firstOutside;
                    sink.invertEnd = (sink.invertBase << 1) - 1;
                    insideBase = sink.invertBase;
                } else {
                    aps5_tess_set_patch(sink, insideOffset, insidePoints[axis] - 1, firstInside, outsidePoints[edge], outsideOffset, firstOutside);
                    insideBase = 0;
                    outsideBase = sink.outsideBase;
                }
            } else if (edge == 2 && ring == degenerate[axis]) {
                sink.patched2 = 1;
                sink.invertBase = insideOffset;
                sink.cornerBad = -1;
                sink.cornerReplacement = -1;
                sink.invertEnd = sink.invertBase << 1;
                insideBase = sink.invertBase;
            }
            aps5_tess_stitch(sink, out, triangles, ring == 1, insidePoints[axis], insideBase, patch.inside[axis].halfPoints, patch.insideParity[axis], outsideBase, outsideHalf[edge], outsideParity[edge]);
            sink.patched = 0;
            sink.patched2 = 0;
            outsideOffset += outsidePoints[edge] - 1;
            if (edge == 2 && ring == degenerate[axis]) insideOffset -= insidePoints[axis] - 1;
            else insideOffset += insidePoints[axis] - 1;
            outsidePoints[edge] = insidePoints[axis];
        }
        if (ring == 1) {
            for (int edge = 0; edge < 4; ++edge) {
                outsideHalf[edge] = patch.inside[edge & 1].halfPoints;
                outsideParity[edge] = patch.insideParity[edge & 1];
            }
        }
    }
    const int u = patch.insidePoints[0];
    const int v = patch.insidePoints[1];
    if (u > v && patch.insideParity[1] != 0) {
        const int quads = (((u >> 1) - (v >> 1)) << 1) + (patch.insideParity[0] == 0 ? 2 : 1);
        if (aps5_tess_skip(sink, quads * 2)) return;
        const int before = sink.next;
        sink.patched2 = 1;
        sink.invertBase = outsideOffset + quads + 2;
        sink.cornerBad = sink.invertBase;
        sink.cornerReplacement = outsideOffset;
        sink.invertEnd = sink.invertBase + sink.invertBase + quads;
        aps5_tess_stitch_regular(sink, out, false, 0, quads + 1, sink.invertBase, outsideOffset + 1);
        sink.patched2 = 0;
        sink.next = before + quads * 2;
    } else if (v >= u && patch.insideParity[0] != 0) {
        const int quads = (((v >> 1) - (u >> 1)) << 1) + (patch.insideParity[1] == 0 ? 2 : 1);
        if (aps5_tess_skip(sink, quads * 2)) return;
        const int before = sink.next;
        sink.patched2 = 1;
        sink.invertBase = outsideOffset + quads + 1;
        sink.cornerBad = -1;
        sink.invertEnd = sink.invertBase + sink.invertBase + quads;
        aps5_tess_stitch_regular(sink, out, false, patch.insideParity[1] == 0 ? 0 : 1, quads + 1, sink.invertBase, outsideOffset);
        sink.patched2 = 0;
        sink.next = before + quads * 2;
    }
}

APS5_TESS_FN int aps5_tess_corners(APS5_TESS_THREAD const Aps5TessPatch& patch) {
    if (patch.topology == 0) return 1;
    if (patch.topology == 1) return 2;
    return 3;
}

// Triangles of a tri or quad patch: writes the corners of triangles [first, first + count) to out
// (three point indices each) and returns the patch's triangle count.
APS5_TESS_FN int aps5_tess_triangles(APS5_TESS_THREAD const Aps5TessPatch& patch, int first, int count, APS5_TESS_OUT int* out) {
    Aps5TessSink sink;
    sink.next = 0;
    sink.first = first;
    sink.end = first + count;
    sink.clockwise = patch.topology == 2 ? 1 : 0;
    sink.patched = 0;
    sink.insideDelta = 0;
    sink.insideBad = 0;
    sink.insideReplacement = 0;
    sink.outsideBase = 0;
    sink.outsideDelta = 0;
    sink.outsideBad = 0;
    sink.outsideReplacement = 0;
    sink.patched2 = 0;
    sink.invertBase = 0;
    sink.invertEnd = 0;
    sink.cornerBad = -1;
    sink.cornerReplacement = -1;
    if (patch.minimum != 0) {
        if (patch.domain == 1) {
            aps5_tess_triangle(sink, out, 0, 1, 2);
        } else {
            aps5_tess_triangle(sink, out, 0, 1, 3);
            aps5_tess_triangle(sink, out, 1, 2, 3);
        }
    } else if (patch.domain == 1) {
        aps5_tess_triangle_rings(patch, sink, out);
    } else {
        aps5_tess_quad_rings(patch, sink, out);
    }
    return sink.next;
}

// Primitives (points, lines or triangles by the output topology) of a patch.
APS5_TESS_FN int aps5_tess_primitive_count(APS5_TESS_THREAD const Aps5TessPatch& patch, APS5_TESS_OUT int* unused) {
    if (patch.culled != 0) return 0;
    if (patch.topology == 0) return patch.points;
    if (patch.topology == 1) return patch.domain == 0 ? patch.lines * (patch.linePoints - 1) : patch.points - 1;
    return aps5_tess_triangles(patch, 0x7fffffff, 0, unused);
}

// The corners (aps5_tess_corners each) of primitives [first, first + count) into out; returns how
// many of them exist.
APS5_TESS_FN int aps5_tess_window(APS5_TESS_THREAD const Aps5TessPatch& patch, int first, int count, APS5_TESS_OUT int* out) {
    if (patch.culled != 0) return 0;
    if (patch.topology == 0) {
        int written = 0;
        for (int k = first; k < first + count && k < patch.points; ++k) out[written++] = k;
        return written;
    }
    if (patch.topology == 1) {
        int written = 0;
        if (patch.domain == 0) {
            const int segments = patch.linePoints - 1;
            for (int k = first; k < first + count && k < patch.lines * segments; ++k) {
                const int line = k / segments;
                const int point = line * patch.linePoints + (k - line * segments);
                out[written * 2] = point;
                out[written * 2 + 1] = point + 1;
                written += 1;
            }
            return written;
        }
        for (int k = first; k < first + count && k < patch.points - 1; ++k) {
            out[written * 2] = k;
            out[written * 2 + 1] = k + 1;
            written += 1;
        }
        return written;
    }
    const int total = aps5_tess_triangles(patch, first, count, out);
    const int last = total < first + count ? total : first + count;
    return last > first ? last - first : 0;
}

)
