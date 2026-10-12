#!/usr/bin/env python3
"""Fail when code that an arm64 build compiles still contains x86 inline assembly or x86 intrinsics.

Usage: python3 -I tools/mac/check_no_x86.py <build-dir> [--jobs N] [--filter REGEX] [--repo ROOT]

The build directory must have compile_commands.json (CMAKE_EXPORT_COMPILE_COMMANDS, on by default for the
macOS full build). Every repository
translation unit in compile_commands.json (not 3rdparty) is preprocessed with exactly the flags the
build uses, so code that `#if defined(__x86_64__)`, `_WIN32` or CMake keeps off the arm64 build
is not seen. Only lines that come from repository files are scanned (SDK, libc++ and 3rdparty
headers are skipped). Reported: x86 intrinsic calls and types, x86 CPU feature builtins, x86
intrinsic headers that get included, and x86 registers or mnemonics inside asm statements (an
empty `asm volatile("" ::: "memory")` compiler barrier and asm labels are architecture neutral).
Exit status 0 = no findings, 1 = findings, 2 = no findings but a translation unit could not be preprocessed.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

INTRINSIC = re.compile(
    r"\b(_mm(?:256|512)?_\w+|_MM_\w+|__builtin_ia32_\w+|__rdtscp?|_rdtsc|__cpuid(?:ex|_count)?|__get_cpuid\w*"
    r"|_xgetbv|_fx(?:save|rstor)(?:64)?|_xsave\w*|_xrstor\w*|__builtin_cpu_(?:supports|is|init)"
    r"|_BitScan(?:Forward|Reverse)(?:64)?|__read[fg]s(?:byte|word|dword|qword)|_(?:read|write)[fg]sbase_u(?:32|64)"
    r"|_(?:pdep|pext|bzhi|bextr|lzcnt|tzcnt|mulx|addcarryx?|subborrow|rdrand|rdseed)_u(?:16|32|64)"
    r"|__m(?:64|128i|128d|256|256i|256d|512|512i|512d)|YieldProcessor|_ReadWriteBarrier|__debugbreak)\b")
X86_HEADER = re.compile(
    r'/(?:x86|ia32|imm|xmm|emm|pmm|tmm|smm|nmm|wmm|amm|mm|avx\w*|bmi\w*|fma\w*|f16c|lzcnt|popcnt|x86gpr|ammintrin|cet|'
    r'cldemote|clflushopt|clwb|clzero|fxsr|xsave\w*|rdseed|rtm|shaintrin|wbnoinvd|waitpkg|movdir\w*|pconfig|pku|ptwrite|sgx|serialize|tsxldtrk|vaes|vpclmulqdq|gfni)intrin\.h"'
    r'|/cpuid\.h"|/mm_malloc\.h"')
X86_ASM = re.compile(
    r"%%?(?:[re]?[abcd]x|[re]?[sd]il?|[re]?[sb]pl?|r(?:8|9|1[0-5])[dwb]?|[abcd][lh]|[xyz]mm\d+|k[0-7]|rip|eflags|rflags|[cdefgs]s|st(?:\(\d\))?)\b"
    r"|\b(?:pushq|popq|pushfq|popfq|movq|movl|movw|movb|movabsq|movaps|movups|movdqa|movdqu|leaq|leal|callq|retq|jmpq|xorl|xorq|testl|testq|cmpl|cmpq|addq|subq|andq|orq|incl|decl|"
    r"fxsave64|fxrstor64|fxsave|fxrstor|xsave64|xrstor64|fnstcw|fldcw|fnstsw|fninit|ldmxcsr|stmxcsr|vzeroall|vzeroupper|rdtscp?|cpuid|xgetbv|ud2|lfence|sfence|mfence|pause|int3|"
    r"rdfsbase|wrfsbase|rdgsbase|wrgsbase|swapgs|cmpxchg16b|lock)\b"
    r"|\.(?:intel_syntax|att_syntax|code64|code32)\b")
X86_REGISTER = re.compile(
    r"^(?:[re][abcd]x|[re][sd]i|[re][sb]p|r(?:8|9|1[0-5])[dwb]?|[xyz]mm(?:\d|[12]\d|3[01])|k[0-7]|st(?:\([0-7]\))?|eflags|rflags|dirflag)$")
ASM_START = re.compile(r"\b(?:__asm__|__asm|asm)\b(?:\s|volatile\b|__volatile__\b|inline\b|goto\b)*\(")
LINE_MARKER = re.compile(r'^#\s+(\d+)\s+"((?:[^"\\]|\\.)*)"')
STRING = re.compile(r'"((?:[^"\\\n]|\\.)*)"')
DIAGNOSTIC = re.compile(r": (?:fatal error|error|warning):")


def preprocess_command(entry):
    args = shlex.split(entry["command"]) if "command" in entry else list(entry["arguments"])
    out, skip = [], False
    for argument in args:
        if skip:
            skip = False
            continue
        if argument == "-o":
            skip = True
            continue
        if argument == "-c":
            continue
        out.append(argument)
    return out + ["-E", "-o", "-"]


def scan(entry, repo, third_party):
    is_assembly = entry["file"].endswith((".S", ".s", ".asm"))
    try:
        result = subprocess.run(preprocess_command(entry), cwd=entry["directory"], capture_output=True, text=True,
                                errors="replace", timeout=600)
    except subprocess.TimeoutExpired:
        return entry["file"], None, "timeout"
    findings = set()
    error = None
    if result.returncode != 0:
        # An x86 intrinsic header stops with #error on arm64; report the repository file that included it.
        chain = []
        for line in result.stderr.splitlines():
            included = re.match(r"In file included from (.+?):(\d+):", line)
            if included:
                chain.append((os.path.normpath(os.path.join(entry["directory"], included.group(1))), int(included.group(2))))
                continue
            if "only meant to be used on x86" in line:
                ours = [link for link in chain if link[0].startswith(repo) and not link[0].startswith(third_party)]
                site = (ours[-1][0], ours[-1][1], "x86 intrinsic header (#error on arm64)") if ours else None
                if site and not any(finding[:3] == site for finding in findings):
                    findings.add(site + (os.path.basename(line.split(":")[0]),))
            if DIAGNOSTIC.search(line) and ": warning:" not in line and error is None and "only meant to be used on x86" not in line:
                error = line.strip()
            if not line.startswith(" ") and not included:
                chain = [] if DIAGNOSTIC.search(line) else chain
        if error is None and not findings:
            error = f"the preprocessor exited with status {result.returncode}"
    current, number, ours = "", 0, False
    kept = []  # (file, line, text) of repository lines, in order
    for raw in result.stdout.split("\n"):
        marker = LINE_MARKER.match(raw)
        if marker:
            entering = " 1" in raw[marker.end():].split("//")[0]
            if entering and ours and X86_HEADER.search(raw):  # a repository file includes an x86 intrinsic header
                findings.add((current, number, "x86 intrinsic header", os.path.basename(marker.group(2))))
            number, current = int(marker.group(1)), os.path.normpath(os.path.join(entry["directory"], marker.group(2)))
            ours = current.startswith(repo) and not current.startswith(third_party)
            continue
        if ours and raw.strip():
            kept.append((current, number, raw))
            if not is_assembly:
                for match in INTRINSIC.finditer(raw):
                    findings.add((current, number, "x86 intrinsic", match.group(0)))
            elif X86_ASM.search(raw):
                findings.add((current, number, "x86 assembly", raw.strip()[:160]))
        number += 1
    if not is_assembly:
        text = "\n".join(line for _, _, line in kept)
        starts, offset = [], 0
        for index, (_, _, line) in enumerate(kept):
            starts.append(offset)
            offset += len(line) + 1

        def origin(position):
            low, high = 0, len(starts) - 1
            while low < high:
                middle = (low + high + 1) // 2
                if starts[middle] <= position:
                    low = middle
                else:
                    high = middle - 1
            return kept[low][0], kept[low][1]

        for start in ASM_START.finditer(text):
            depth, position, body = 1, start.end(), []
            while position < len(text) and depth:
                character = text[position]
                if character == '"':
                    literal = STRING.match(text, position)
                    if literal:
                        body.append(literal.group(1))
                        position = literal.end()
                        continue
                depth += character == "("
                depth -= character == ")"
                position += 1
            hit = X86_ASM.search("\n".join(body))
            # Operands and clobbers name registers without '%': "xmm0", "rax", "st(7)".
            register = next((literal for literal in body if X86_REGISTER.match(literal)), None)
            if hit or register:
                file, line = origin(start.start())
                findings.add((file, line, "x86 inline assembly", hit.group(0) if hit else register))
    return entry["file"], findings, error


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("build")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--filter", default="", help="only scan translation units whose path matches this regex")
    parser.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    args = parser.parse_args()
    repo = os.path.normpath(args.repo) + os.sep
    third_party = os.path.join(repo, "3rdparty") + os.sep
    with open(os.path.join(args.build, "compile_commands.json"), encoding="utf-8") as handle:
        entries = json.load(handle)
    selected = [entry for entry in entries
                if os.path.normpath(entry["file"]).startswith(repo) and not os.path.normpath(entry["file"]).startswith(third_party)
                and re.search(args.filter, entry["file"])]
    unique, seen = [], set()
    for entry in selected:  # a source compiled into several targets with the same flags is scanned once
        key = (entry["file"], re.sub(r"\s-o\s+\S+", "", entry.get("command", " ".join(entry.get("arguments", [])))))
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    selected = unique
    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(lambda entry: scan(entry, repo, third_party), selected))
    findings, failed = set(), []
    for file, found, error in results:
        if found:
            findings |= found
        if error is not None and file not in {name for name, _ in failed}:
            failed.append((file, error))
    for file, line, kind, text in sorted(findings):
        print(f"{os.path.relpath(file, repo)}:{line}: {kind}: {text}")
    for file, error in failed:
        print(f"error: cannot preprocess {os.path.relpath(file, repo)}: {error}", file=sys.stderr)
    print(f"{len(selected)} translation unit(s) scanned, {len(findings)} finding(s), {len(failed)} not preprocessed")
    sys.exit(1 if findings else 2 if failed else 0)


if __name__ == "__main__":
    main()
