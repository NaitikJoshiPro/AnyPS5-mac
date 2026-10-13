// Stand-in for Mesa's util/macros.h: tessellator.cpp only uses UNREACHABLE (in CHLSLTessellator).
#pragma once
#include <cstdlib>
#define UNREACHABLE(message) std::abort()
