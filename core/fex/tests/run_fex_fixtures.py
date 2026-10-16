"""Relink the programs in fex_fixtures.py, and the C++ programs of the macOS fixtures when the tools
in fex_toolchain.py are there, and run them with aps5-fex.

    python3 run_fex_fixtures.py <relinker> <aps5-fex> <patched prx directory> [--only <fixture>...]

Each program reports its result through its exit status and standard output."""
import os
import re
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fex_fixtures
import fex_toolchain
from nid import nid

MACOS = pathlib.Path(__file__).resolve().parents[2] / "relinker" / "relinker" / "tests" / "macos"
HERE = pathlib.Path(__file__).resolve().parent

EXPECTED = {
    "hello": (42, "hello from x86-64 guest code on arm64\n"),
    "floating": (41, ""),
    "sorting": (19, ""),
    "variadic": (0, "42 x 2.50 4 5 6\nprinted 7\n"),
    "opening": (0, None),
    "threading": (42, None),
    "extended": (42, ""),
}


def created_with_mode(directory):
    path = directory / "created.txt"
    if not path.exists():
        return "created.txt was not created"
    mode = path.stat().st_mode & 0o777
    expected = 0o640 & ~current_umask()
    return None if mode == expected else f"created.txt has mode {mode:o}, expected {expected:o}"


def current_umask():
    mask = os.umask(0)
    os.umask(mask)
    return mask


def sampled_spin(directory):
    """The sampler's report names the guest loop and the polled address (pc_sample.cpp)."""
    # Global symbols carry their NIDs (fex_toolchain.nidify).
    poll, poll_size = fex_toolchain.symbol(directory / "input.elf", nid("sample_poll"))
    flag, _ = fex_toolchain.symbol(directory / "input.elf", nid("sampleFlag"))
    lines = (directory / "stderr.txt").read_text().splitlines()
    rows = [line.split() for line in lines if line.startswith("[fex-sample]") and "% rip 0x" in line]
    if not rows:
        return "no [fex-sample] rows in stderr: " + " | ".join(lines[-5:])
    # "[fex-sample] 97.0% rip 0x... eboot.elf+0x... jit", then the register line that follows it.
    best = max(range(len(rows)), key=lambda i: float(rows[i][1].rstrip("%")))
    rip = int(rows[best][3], 16)
    offset = int(rows[best][4].rsplit("+", 1)[1], 16)
    base = rip - offset
    if not poll <= offset < poll + poll_size or rows[best][-1] != "jit":
        return f"hottest row {' '.join(rows[best])} is not in sample_poll [{poll:#x}, {poll + poll_size:#x})"
    index = [i for i, line in enumerate(lines) if line.startswith("[fex-sample]") and "% rip 0x" in line][best]
    values = dict(item.split("=") for item in lines[index + 1].split()[1:] + lines[index + 2].split()[1:])
    if int(values["rdi"], 16) != base + flag:
        return f"rdi={values['rdi']} at the hot RIP, expected sampleFlag at {base + flag:#x}"
    return None


def sites_forgotten(directory):
    """With APS5_FEX_CLEAR_CODE=1 the freed code buffers must release their displaced-access sites."""
    text = (directory / "stderr.txt").read_text(errors="replace")
    forgotten = [int(m) for m in re.findall(r"\[displaced\] code freed .*?: (\d+) sites forgotten", text)]
    if os.environ.get("APS5_FEX_CLEAR_CODE") == "1" and not any(forgotten):
        return "no freed code buffer released displaced-access sites (expected '[displaced] code freed ...: N sites forgotten')"
    return None


CHECKS = {"opening": created_with_mode, "pc-sample": sampled_spin, "displaced-churn": sites_forgotten}
ENVIRONMENT = {"pc-sample": {"APS5_FEX_SAMPLE": "1", "APS5_FEX_SAMPLE_REPORT": "60"},
               "displaced-churn": {"APS5_DISPLACED_TRACE": "1"}}


def compiled_executable(directory, sources, libraries, extra=(), tpoff=False):
    """Builds input.elf from C and C++ sources whose imports come from the named stub libraries, or
    from libc.prx when libraries is None; with tpoff, its initial-exec TLS keeps TPOFF64 relocations."""
    objects = []
    imported, defined = set(), set()
    for source in sources:
        obj = directory / (pathlib.Path(source).stem + ".o")
        fex_toolchain.compile(MACOS / source, obj, pic=tpoff, extra=(["-x", "c"] if source.endswith(".c") else []) + list(extra))
        imported.update(fex_toolchain.symbols(obj, "-u"))
        defined.update(fex_toolchain.symbols(obj, "-g", "--defined-only"))
        objects.append(obj)
    stubs = []
    for name, names in (libraries or {"libc.prx": sorted(imported - defined)}).items():
        fex_toolchain.stub_library(names, directory / name, name)
        stubs.append(directory / name)
    renamed = []
    for obj in objects:
        fex_toolchain.nidify(obj, obj.with_suffix(".nid.o"))
        renamed.append(obj.with_suffix(".nid.o"))
    (fex_toolchain.link_tpoff_executable if tpoff else fex_toolchain.link_executable)(renamed, stubs, directory / "input.elf")
    fex_toolchain.make_header_room(directory / "input.elf")


def with_exports(directory, sources, libraries):
    """Like compiled_executable (linked as a shared object, so its default-visibility symbols are in .dynsym),
    with the PS5 tag that makes the relinker keep them as exports."""
    compiled_executable(directory, sources, libraries, tpoff=True)
    fex_toolchain.export_symbols(directory / "input.elf")


def with_process_parameters(directory, sources, libraries, extra=()):
    """Like compiled_executable, with PT_SCE_PROCPARAM over the program's processParameters."""
    compiled_executable(directory, sources, libraries, extra)
    fex_toolchain.add_process_parameters(directory / "input.elf")


def compiled_modules(directory, modules, main, libraries):
    """Builds input/eboot.elf from main and input/sce_module from modules, a list of (source, module
    name, the earlier modules it links against), with stub libraries for the imports."""
    staged = directory / "input"
    (staged / "sce_module").mkdir(parents=True)
    stubs = []
    for name, names in libraries.items():
        fex_toolchain.stub_library(names, directory / name, name)
        stubs.append(directory / name)
    built = {}
    for source, name, needed in modules:
        obj = directory / (pathlib.Path(source).stem + ".o")
        fex_toolchain.compile(MACOS / source, obj, pic=True)
        fex_toolchain.nidify(obj, obj.with_suffix(".nid.o"))
        built[name] = staged / "sce_module" / name
        fex_toolchain.link_module([obj.with_suffix(".nid.o")], [built[module] for module in needed] + stubs, built[name], name)
    obj = directory / "main.o"
    fex_toolchain.compile(MACOS / main, obj)
    fex_toolchain.nidify(obj, directory / "main.nid.o")
    fex_toolchain.link_executable([directory / "main.nid.o"], list(reversed(built.values())) + stubs, staged / "eboot.elf")
    fex_toolchain.make_header_room(staged / "eboot.elf")


def with_abi_library(directory, source, libraries, extra=()):
    """Like compiled_executable, plus the test library libAps5BridgeAbi.prx staged beside the build's
    libraries (directory/libs: links to each of them and to the test library)."""
    compiled_executable(directory, [str(HERE / source)], {**libraries, "libAps5BridgeAbi.prx": ABI_EXPORTS}, extra)
    staged = directory / "libs"
    staged.mkdir()
    for library in LIBRARIES.iterdir():
        (staged / library.name).symlink_to(library)
    (staged / "libAps5BridgeAbi.prx").symlink_to(RUNNER.parent / "abi-libs" / "libAps5BridgeAbi.prx")


ABI_EXPORTS = ["Aps5AbiHfa", "Aps5AbiStack", "Aps5AbiMixed", "Aps5AbiLarge", "Aps5AbiLongDouble", "Aps5AbiInt128", "Aps5AbiPosition",
               "Aps5AbiManyDoubles", "Aps5AbiCallInts", "Aps5AbiCallShaped", "Aps5AbiCallLarge", "Aps5AbiCallLongDouble",
               "Aps5AbiStore", "Aps5AbiLoad", "Aps5AbiCallStored", "Aps5AbiMainThreadProbe"]
LIBRARIES = RUNNER = None

COMPILED = {
    "abi-registers": (lambda d: with_abi_library(d, "abi_registers.cpp", {"libc.prx": ["exit"]}), 43),
    "abi-stack": (lambda d: with_abi_library(d, "abi_stack.cpp", {"libc.prx": ["exit"],
                  "libSceAgc.prx": ["sceAgcCbReleaseMem", "sceAgcDcbDmaData"]}), 43),
    "abi-memory": (lambda d: with_abi_library(d, "abi_memory.cpp", {"libc.prx": ["exit", "_ZSt17current_exceptionv"]}), 43),
    "abi-long-double": (lambda d: with_abi_library(d, "abi_long_double.cpp", {"libc.prx": ["exit", "acosl", "atanl", "coshl", "cosl",
                        "exp2l", "expl", "frexpl", "ldexpl", "log10l", "logl", "powl", "sinhl", "sinl", "tanhl", "tanl", "truncl",
                        "strtold", "wcstold", "__signbitl"]}), 43),
    "abi-callbacks": (lambda d: with_abi_library(d, "abi_callbacks.cpp", {"libc.prx": ["exit"]}), 43),
    "main-thread": (lambda d: with_abi_library(d, "main_thread.cpp", {"libc.prx": ["exit"]}), 43),
    "context": (lambda d: compiled_executable(d, [str(HERE / "context.cpp")], {"libc.prx": ["exit"], "libkernel.prx": ["getcontext",
                "setcontext", "sigsetjmp", "siglongjmp"], "libSceLibcInternal.prx": ["makecontext"]}), 43),
    "context-exit": (lambda d: compiled_executable(d, [str(HERE / "context.cpp")], {"libc.prx": ["exit"], "libkernel.prx": ["getcontext",
                     "setcontext", "sigsetjmp", "siglongjmp"], "libSceLibcInternal.prx": ["makecontext"]}, ["-DCONTEXT_EXIT"]), 0),
    "raise": (lambda d: compiled_executable(d, [str(HERE / "raise.cpp")], {"libc.prx": ["exit"], "libkernel.prx": [
              "sceKernelInstallExceptionHandler", "sceKernelRemoveExceptionHandler", "sceKernelRaiseException", "scePthreadSelf"]}), 43),
    "abi-variadic": (lambda d: compiled_executable(d, [str(HERE / "abi_variadic.cpp")], {"libc.prx": ["exit", "snprintf", "swprintf"]}), 43),
    "exception": (lambda d: compiled_executable(d, ["exception.cpp"], None), 43),
    "c-cleanup": (lambda d: compiled_executable(d, ["c_cleanup.c", "c_cleanup_main.cpp"], None), 47),
    "threads": (lambda d: compiled_executable(d, ["threads.cpp"], {"libc.prx": ["exit"],
                "libkernel.prx": ["scePthreadCreate", "scePthreadJoin"]}), 51),
    "jump": (lambda d: compiled_executable(d, [str(HERE / "jump.cpp")],
             {"libc.prx": ["setjmp", "longjmp", "exit"]}), 43),
    "fibers": (lambda d: compiled_executable(d, [str(HERE / "fibers.cpp")],
               {"libc.prx": ["exit"], "libSceFiber.prx": ["_sceFiberInitializeImpl", "sceFiberRun", "sceFiberSwitch",
                "sceFiberReturnToThread"]}), 43),
    "fibers-control": (lambda d: compiled_executable(d, [str(HERE / "fibers_control.cpp")],
                       {"libc.prx": ["exit"], "libSceFiber.prx": ["_sceFiberInitializeImpl", "sceFiberRun", "sceFiberSwitch",
                        "sceFiberReturnToThread"]}), 43),
    "zen2": (lambda d: compiled_executable(d, [str(HERE / "zen2.cpp")],
             {"libc.prx": ["exit"], "libkernel.prx": ["mmap", "munmap"]}, ["-march=znver2", "-O2"]), 43),
    "heap": (lambda d: with_process_parameters(d, [str(HERE / "heap.cpp")],
             {"libc.prx": ["_init_env", "malloc", "free", "calloc", "realloc", "memalign", "posix_memalign", "exit"]}, ["-O2"]), 43),
    "callbacks": (lambda d: compiled_executable(d, [str(HERE / "callbacks.cpp")],
                  {"libc.prx": ["qsort", "bsearch", "strcmp", "exit"]}, ["-O2"]), 43),
    "floats": (lambda d: compiled_executable(d, [str(HERE / "floats.cpp")], {"libc.prx": ["exit"]},
               ["-march=znver2", "-O2", "-frounding-math"]), 43),
    "crypto": (lambda d: compiled_executable(d, [str(HERE / "crypto.cpp")], {"libc.prx": ["exit"]}, ["-march=znver2", "-O2"]), 43),
    "backports": (lambda d: compiled_executable(d, [str(HERE / "backports.cpp")], {"libc.prx": ["exit"]}, ["-march=znver2"]), 43),
    "module": (lambda d: compiled_modules(d, [("greet.cpp", "libgreet.prx", [])], "module_main.cpp",
               {"libc.prx": ["puts", "exit", "memset", "__tls_get_addr"], "libkernel.prx": ["sceKernelGetModuleInfoForUnwind"]}), 143),
    "dynamic-module": (lambda d: compiled_modules(d, [("greet.cpp", "libgreet.prx", [])],
                       HERE / "dynamic_module.cpp",
                       {"libc.prx": ["puts", "exit", "memset", "__tls_get_addr"],
                        "libkernel.prx": ["sceKernelLoadStartModule", "sceKernelDlsym"]}), 43),
    "tls-modules": (lambda d: compiled_modules(d, [("tls_owner.cpp", "libowner.prx", []), ("tls_user.cpp", "libuser.prx", ["libowner.prx"])],
                    "tls_modules_main.cpp", {"libc.prx": ["exit", "__tls_get_addr"],
                    "libkernel.prx": ["scePthreadCreate", "scePthreadJoin"]}), 47),
    "tls-exec": (lambda d: compiled_executable(d, [str(HERE / "tls_exec.cpp")], {"libc.prx": ["exit"],
                 "libkernel.prx": ["scePthreadCreate", "scePthreadJoin"]}, tpoff=True), 43),
    "tls-exec-dynamic": (lambda d: compiled_executable(d, [str(HERE / "tls_exec_dynamic.cpp")], {"libc.prx": ["exit"],
                         "libkernel.prx": ["scePthreadCreate", "scePthreadJoin", "__tls_get_addr"]}, tpoff=True), 43),
    "image-protect": (lambda d: compiled_executable(d, [str(HERE / "image_protect.cpp")], {"libc.prx": ["exit"],
                      "libkernel.prx": ["sceKernelMprotect", "sceKernelQueryMemoryProtection"]}), 43),
    "unaligned": (lambda d: compiled_executable(d, [str(HERE / "unaligned.cpp")], {"libc.prx": ["exit"],
                  "libkernel.prx": ["scePthreadCreate", "scePthreadJoin"]}, ["-O2"]), 43),
    "code-churn": (lambda d: compiled_executable(d, [str(HERE / "code_churn.cpp")], {"libc.prx": ["exit"],
                   "libkernel.prx": ["scePthreadCreate", "scePthreadJoin"]}, ["-O2"]), 43),
    "console-syscall": (lambda d: compiled_executable(d, [str(HERE / "console_syscall.cpp")], {"libc.prx": ["exit"]}), 43),
    "displaced": (lambda d: compiled_executable(d, [str(HERE / "displaced.cpp")], {"libc.prx": ["exit", "memset", "memcpy"],
                  "libkernel.prx": ["sceKernelAllocateDirectMemory", "sceKernelMapDirectMemory", "sceKernelMunmap"]}, ["-O2"]), 43),
    "displaced-churn": (lambda d: compiled_executable(d, [str(HERE / "displaced_churn.cpp")], {"libc.prx": ["exit"],
                        "libkernel.prx": ["sceKernelAllocateDirectMemory", "sceKernelMapDirectMemory", "scePthreadCreate",
                        "scePthreadJoin"]}, ["-O2"]), 43),
    "displaced-unaligned": (lambda d: compiled_executable(d, [str(HERE / "displaced_unaligned.cpp")], {"libc.prx": ["exit"],
                            "libkernel.prx": ["sceKernelAllocateDirectMemory", "sceKernelMapDirectMemory", "scePthreadCreate",
                            "scePthreadJoin"]}, ["-O2"]), 43),
    "guest-guards": (lambda d: with_exports(d, [str(HERE / "guest_guards.cpp")], {"libc.prx": ["exit"],
                     "libkernel.prx": ["scePthreadCreate", "scePthreadJoin"]}), 43),
    "cpuid-brand": (lambda d: compiled_executable(d, [str(HERE / "cpuid_brand.cpp")], {"libc.prx": ["exit"]}), 43),
    "kernel-sys": (lambda d: compiled_executable(d, [str(HERE / "kernel_sys.cpp")], {"libc.prx": ["exit"],
                   "libkernel.prx": ["pathconf"]}), 43),
    "pc-sample": (lambda d: compiled_executable(d, [str(HERE / "pc_sample.cpp")], {"libc.prx": ["exit"]}, ["-O2"]), 43),
    "dlsym-host": (lambda d: compiled_executable(d, [str(HERE / "dlsym_host.cpp")], {"libc.prx": ["exit", "strlen",
                   "_ZSt13set_terminatePFvvE", "_ZSt13get_terminatev"], "libkernel.prx": ["sceKernelDlsym"]}), 43),
}


def main():
    global LIBRARIES, RUNNER
    relinker, runner, libraries = (pathlib.Path(argument).resolve() for argument in sys.argv[1:4])
    LIBRARIES, RUNNER = libraries, runner
    only = set(sys.argv[sys.argv.index("--only") + 1:]) if "--only" in sys.argv else None
    failures = []
    cases = {name: (lambda d, name=name: (d / "input.elf").write_bytes(fex_fixtures.FIXTURES[name]()), status, output)
             for name, (status, output) in EXPECTED.items()}
    if fex_toolchain.available():
        cases.update({name: (build, status, None) for name, (build, status) in COMPILED.items()})
    else:
        print("compiled programs skipped: clang, llvm-nm or an LLVM ELF linker is missing")
    if only is not None:
        unknown = only - cases.keys()
        if unknown:
            sys.exit(f"unknown fixtures: {', '.join(sorted(unknown))}")
        cases = {name: case for name, case in cases.items() if name in only}
    for name, (build, status, output) in cases.items():
        with tempfile.TemporaryDirectory(prefix=f"aps5-fex-{name}-") as directory:
            directory = pathlib.Path(directory)
            build(directory)
            modules = (directory / "input" / "sce_module").exists()
            source = "input/eboot.elf" if modules else "input.elf"
            relinked = subprocess.run([str(relinker), *([] if modules else ["--skip-sce-module"]), source, "eboot.elf"],
                                      cwd=directory, capture_output=True, text=True, timeout=60)
            if relinked.returncode != 0:
                failures.append(f"{name}: relink failed: {relinked.stderr.strip()}")
                continue
            if not (directory / "libs").exists():
                (directory / "libs").symlink_to(libraries, target_is_directory=True)
            executed = subprocess.run([str(runner), "eboot.elf"], cwd=directory, capture_output=True, text=True, timeout=60,
                                      env={**os.environ, **ENVIRONMENT.get(name, {})})
            (directory / "stderr.txt").write_text(executed.stderr)
            if executed.returncode != status or (output is not None and executed.stdout != output):
                failures.append(f"{name}: exit {executed.returncode}, expected {status}; output {executed.stdout!r}, "
                                f"expected {output!r}; {executed.stderr.strip()[-400:]}")
            elif name in CHECKS and (problem := CHECKS[name](directory)):
                failures.append(f"{name}: {problem}")
    for failure in failures:
        print(failure)
    if failures:
        sys.exit(1)
    print(f"{len(cases)} guest programs ran through FEXCore")


if __name__ == "__main__":
    main()
