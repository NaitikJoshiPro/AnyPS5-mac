#!/usr/bin/env python3
"""Sorts the GPU-labelled tests of a build directory into the M2 bring-up groups (docs/mac/M2_DESIGN.md §8).

Usage:
    python3 -I tools/mac/gpu_groups.py [--build build-metal] [--write | --check | --list GROUP]

The classification reads the build directory only (CTestTestfile.cmake for each test's command,
build.ninja for each test executable's sources) and the sources themselves; it never runs a test.
The committed map tools/mac/gpu_groups.txt is what tools/mac/gpu_suite.sh uses, so pass counts per
group stay comparable between runs; tests the map does not list yet are classified on the fly.
  --write   add the tests the map lacks and drop the ones the build no longer has; the groups of listed tests
            stay (with --reclassify, every test is classified again)
  --check   exit 1 when the map lacks a test or lists one the build no longer has; tests whose sources now
            classify differently are reported without failing
  --list G  print the tests of group G, one per line

Groups:
  G0 CPU-only tests that carry the gpu label (decoders, structurizer, PM4 parsing, recompile-only)
  G1 compute-only execution tests (ALU, LDS, buffers, global/flat memory, 32-bit atomics, wave ops, flow)
  G2 compute tests with images or samplers
  G3 draw tests
  G4 PM4/driver tests (submission through the driver, shutdown, prepared shaders, shared writes)
  G5 presentation (flip, video out, pad): runs headless, gpu_suite forces SDL_VIDEODRIVER=offscreen
  G6 f64 and 64-bit atomics
  G7 Metal-only tests and the Vulkan-internal tests that Metal twins replace

HOLD matches the tests tools/mac/gpu_suite.sh never runs, whatever the map says. It is empty (HELD_TESTS = ()):
agc_driver_lds_atomics64, the last held test, was released on 2026-10-13 after supervised passes. The mechanism
(--rerun-held, the "held" summary column) stays for future use. PRESENTATION matches the G5 tests (flip, present, video out, libSceVideoOut):
they run in normal suite runs, headless, because gpu_suite forces SDL_VIDEODRIVER=offscreen for every test.
tools/mac/gpu_quarantine.txt adds tests that took the machine down (kernel panic, GPU hang) until they are cleared;
a missing or malformed quarantine file stops gpu_suite.sh from running anything (fail closed).
"""
import argparse
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAP = Path(__file__).resolve().parent / "gpu_groups.txt"
GROUPS = ("G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7")

# Tests gpu_suite.sh never runs (CLAUDE.md GPU safety). Empty: agc_driver_lds_atomics64 was released on 2026-10-13
# after supervised passes. GPU_GROUPS_EXTRA_HELD (comma-separated names) adds tests for the unit tests only.
HELD_TESTS: tuple = ()


def hold_pattern(names):
    names = [name for name in names if name]
    return re.compile("^(" + "|".join(re.escape(name) for name in names) + ")$") if names else re.compile(r"(?!)")


HOLD = hold_pattern(HELD_TESTS + tuple(os.environ.get("GPU_GROUPS_EXTRA_HELD", "").split(",")))
# G5 presentation tests: run headless by gpu_suite (SDL_VIDEODRIVER=offscreen), no longer held.
PRESENTATION = re.compile(r"(^|_)(flip|visual|present|presentation|swapchain)(_|$)|^(guest_)?video_out_|^pad_sdl$|^guest_buffer_reuse$")
PRESENTATION_DIRECTORIES = ("core/libs/prx/libSceVideoOut",)
# Tests quarantined after they took the machine down: `<test> | <reason> | <TechnicalDebt anchor>` per line.
QUARANTINE = Path(__file__).resolve().parent / "gpu_quarantine.txt"

# Tests that drive Vulkan internals directly (raw Vulkan or MockVulkan); Metal twins replace them (M2_DESIGN.md §8).
VULKAN_INTERNAL = frozenset("""
agc_pipeline_cache agc_pipeline_library agc_pipeline_library_off agc_pipeline_library_fast_only agc_unchanged_cpu_stamps
agc_storage_baseline_settled agc_storage_upload_lifetime agc_storage_upload_failure agc_host_import_unmap agc_unit_shadow
agc_descriptor_pool_recycling agc_draw_snapshot_cap agc_multisample_guest_access agc_draw_inputs_in_place
agc_driver_graphics agc_driver_bda_device agc_driver_oversized_descriptor_sets agc_driver_pooled_buffer_range
agc_driver_sampled_cache_budget agc_driver_sampled_cache_budget_memory agc_driver_resource_cache_storage_admission
agc_driver_resource_cache_oversized_storage_admission agc_driver_resource_cache_storage_clear
agc_descriptor_pool_reset_failure agc_compute_descriptor_layout_flags
agc_buffer_pool agc_recorder_kept_bytes
agc_descriptor_cache agc_pipeline_journal agc_pipeline_library_clear_waits_for_build agc_pipeline_store_lazy_buffer_pool
agc_recorder_device_selection agc_sampled_storage_base_level_usage agc_draw_descriptors agc_driver_host_import_lifetime
agc_driver_host_import_mutation_race agc_driver_unit_shadow agc_driver_unit_shadow_budget
agc_buffer_pool_small_device_shared agc_buffer_pool_small_device_no_staging
agc_driver_dcc_key_presync agc_driver_resource_cache_growth agc_driver_pipeline_journal
agc_command_batch_failure agc_descriptor_cache_failure
""".split())
# Also Vulkan-internal: the readback variants intercept vkCmdDraw through the Vulkan Graphics::Context test hook.
VULKAN_INTERNAL_FAMILIES = re.compile(r"^agc_driver_pipeline_optimized_readback")

# PM4/driver-level tests named in M2_DESIGN.md §8 (G4) whatever their sources look like.
DRIVER_LEVEL = re.compile(r"^agc_driver$|^agc_driver_pm4|^agc_driver_wait_shutdown$|^agc_driver_resolution_shutdown$|^agc_driver_mesh$"
                          r"|^agc_driver_cmask_fast_clear$|^agc_shared_write|^agc_driver_prepared_|^agc_driver_readonly_resummarize$")

FEATURES = {
    "device": re.compile(r"OpenVulkanTestDevice|\bVulkanDevice\b|MetalDevice\b|MTLCreateSystemDefaultDevice|id<MTLDevice>"),
    "submit": re.compile(r"AgcDriver::Submit\b|\bSubmit\(&|AgcDriverSubmit|sceAgcDriverSubmit|AgcDriverInit|Execution/include/Driver\.hpp\""),
    "draw": re.compile(r"\.Draw\(|->Draw\(|\bDrawIndexed|\bDrawIndirect|\bDrawMesh"),
    "image": re.compile(r"TextureDescriptor|SamplerDescriptor|StorageTexture|\bimage_(load|store|sample|gather|atomic|get_resinfo|bvh)|FillTexture|ImageResources"),
    # Whole-test skips only: tests that skip a few f64 or 64-bit atomic rows stay in their own group.
    "f64": re.compile(r'"skipped, [^"\n]*(Float64|f64)'),
    "atomic64": re.compile(r'"skipped, [^"\n]*Int64Atomics'),
}

ADD_TEST = re.compile(r'^add_test\((.*)\)\s*$')
SET_PROPERTIES = re.compile(r'^set_tests_properties\("([^"]+)" PROPERTIES (.*)\)\s*$')
ARGUMENT = re.compile(r'"((?:\\.|[^"\\])*)"|\[(=*)\[(.*?)\]\2\]|(\S+)')
LINK = re.compile(r'^build (tests/\S+): \S*EXECUTABLE_LINKER\S* (.*?)(?: \||$)')
OBJECT = re.compile(r'(.*)/CMakeFiles/[^/]+\.dir/(.*)\.o$')


def arguments(text):
    found = []
    for match in ARGUMENT.finditer(text):
        for group in (1, 3, 4):
            if match.group(group) is not None:
                found.append(match.group(group))
                break
    return found


def registered_tests(build):
    """{name: {"command": [...], "properties": {...}, "directory": source-relative directory}} from CTestTestfile.cmake."""
    tests = {}
    for testfile in sorted(Path(build).rglob("CTestTestfile.cmake")):
        directory = os.path.relpath(testfile.parent, build)
        for line in testfile.read_text(errors="replace").splitlines():
            added = ADD_TEST.match(line)
            if added:
                found = arguments(added.group(1))
                tests[found[0]] = {"command": found[1:], "properties": {}, "directory": directory}
                continue
            properties = SET_PROPERTIES.match(line)
            if properties and properties.group(1) in tests:
                found = arguments(properties.group(2))
                tests[properties.group(1)]["properties"].update(zip(found[0::2], found[1::2]))
    return tests


def executable(command):
    """The test executable of a command, looking through `cmake -E env ... <exe>` wrappers."""
    if not command:
        return ""
    if os.path.basename(command[0]) in ("cmake", "cmake.exe") and command[1:3] == ["-E", "env"]:
        for argument in command[3:]:
            if not argument.startswith("-") and "=" not in argument:
                return argument
    return command[0]


def executable_sources(build):
    """{executable basename: [repository-relative sources]} from the link lines of build.ninja."""
    sources = {}
    ninja = Path(build) / "build.ninja"
    if not ninja.exists():
        return sources
    for line in ninja.read_text(errors="replace").splitlines():
        link = LINK.match(line)
        if not link:
            continue
        found = []
        for item in link.group(2).split():
            target = OBJECT.match(item)
            if target:
                relative = "/".join(".." if part == "__" else part for part in target.group(2).split("/"))
                found.append(os.path.normpath(os.path.join(target.group(1), relative)))
        sources[os.path.basename(link.group(1))] = found
    return sources


def test_sources(sources):
    return [source for source in sources if "/tests/" in "/" + source]


def classify_one(name, directory, text):
    """The group of one test from its name, its CTest directory and the text of its own test sources."""
    if name == "agc_driver_lds_atomics64":
        return "G6"
    if PRESENTATION.search(name) or any(directory == shown or directory.startswith(shown + "/") for shown in PRESENTATION_DIRECTORIES):
        return "G5"
    if name.startswith("agc_metal_") or name in VULKAN_INTERNAL or VULKAN_INTERNAL_FAMILIES.search(name):
        return "G7"
    feature = {key: bool(pattern.search(text)) for key, pattern in FEATURES.items()}
    if DRIVER_LEVEL.search(name):
        return "G4"
    if not (feature["device"] or feature["submit"] or feature["draw"]):
        return "G0"
    if feature["f64"] or feature["atomic64"]:
        return "G6"
    if feature["submit"]:
        return "G4"
    if feature["draw"]:
        return "G3"
    if feature["image"]:
        return "G2"
    return "G1"


def classify(build, repo=REPO):
    """{test: group} for every test of the build directory that carries the gpu label."""
    tests = registered_tests(build)
    sources = executable_sources(build)
    cache = {}

    def read(path):
        if path not in cache:
            full = Path(repo) / path
            cache[path] = full.read_text(errors="replace") if full.is_file() else ""
        return cache[path]

    groups = {}
    for name, test in tests.items():
        if "gpu" not in test["properties"].get("LABELS", "").split(";"):
            continue
        own = test_sources(sources.get(os.path.basename(executable(test["command"])), []))
        groups[name] = classify_one(name, test["directory"].replace(os.sep, "/"), "\n".join(read(source) for source in own))
    return groups


def load_map(path=MAP):
    groups = {}
    if not Path(path).exists():
        return groups
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 2 or fields[1] not in GROUPS:
            raise ValueError(f"{path}:{number}: expected '<test> <G0..G7>', got {line!r}")
        groups[fields[0]] = fields[1]
    return groups


def write_map(groups, path=MAP):
    lines = ["# M2 GPU test groups (docs/mac/M2_DESIGN.md §8), written by tools/mac/gpu_groups.py --write.",
             "# <test> <group>; gpu_suite.sh never runs the tests gpu_groups.HOLD matches, whatever their group (none at present: agc_driver_lds_atomics64 was released 2026-10-13; G5 runs headless)."]
    for group in GROUPS:
        lines += [f"{name} {group}" for name in sorted(name for name, found in groups.items() if found == group)]
    Path(path).write_text("\n".join(lines) + "\n")


def load_quarantine(path=QUARANTINE):
    """{test: reason} of the quarantine file; raises (OSError, ValueError) when it is missing or malformed, so a
    caller that runs GPU tests fails closed instead of running a quarantined test."""
    quarantined = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) != 3 or not all(fields) or not re.fullmatch(r"[A-Za-z0-9_.+-]+", fields[0]):
            raise ValueError(f"{path}:{number}: expected '<test> | <reason> | <TechnicalDebt anchor>', got {line!r}")
        quarantined[fields[0]] = f"{fields[1]} ({fields[2]})"
    return quarantined


_quarantine_cache = None


def quarantine_reason(name):
    """Why `name` is quarantined, or None. Raises like load_quarantine when the file cannot be read."""
    global _quarantine_cache
    if _quarantine_cache is None:
        _quarantine_cache = load_quarantine()
    return _quarantine_cache.get(name)


def held(name):
    return bool(HOLD.search(name)) or quarantine_reason(name) is not None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--build", default="build-metal", help="CMake build directory (default: build-metal)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="update tools/mac/gpu_groups.txt (new tests added, removed ones dropped)")
    mode.add_argument("--check", action="store_true", help="fail when the map is out of date")
    mode.add_argument("--list", choices=GROUPS, help="print the tests of one group")
    parser.add_argument("--reclassify", action="store_true", help="with --write: classify every test again")
    args = parser.parse_args()
    build = Path(args.build)
    if not build.is_absolute():
        build = Path.cwd() / build
    if not (build / "CTestTestfile.cmake").exists():
        parser.error(f"{build} is not a configured build directory")
    current = classify(build)
    stored = load_map()
    if args.write:
        merged = dict(current) if args.reclassify else {name: stored.get(name, group) for name, group in current.items()}
        write_map(merged)
        print(f"wrote {MAP.relative_to(REPO)}: {len(merged)} tests, {len(set(current) - set(stored))} new, {len(set(stored) - set(current))} dropped")
        return 0
    if args.check:
        changes = [f"new {name} {current[name]}" for name in sorted(set(current) - set(stored))]
        changes += [f"removed {name} {stored[name]}" for name in sorted(set(stored) - set(current))]
        for change in changes:
            print(change)
        for name in sorted(set(stored) & set(current)):
            if stored[name] != current[name]:
                print(f"note: {name} is in {stored[name]}, its sources now classify as {current[name]} (--write --reclassify moves it)")
        if changes:
            print(f"{len(changes)} difference(s): run tools/mac/gpu_groups.py --write and review the diff")
            return 1
        print(f"{MAP.name} is up to date ({len(current)} tests)")
        return 0
    effective = {name: stored.get(name, group) for name, group in current.items()}
    if args.list:
        for name in sorted(name for name, group in effective.items() if group == args.list):
            print(name)
        return 0
    for group in GROUPS:
        names = [name for name, found in effective.items() if found == group]
        print(f"{group} {len(names):4d} tests, {sum(held(name) for name in names)} held back")
    return 0


if __name__ == "__main__":
    sys.exit(main())
