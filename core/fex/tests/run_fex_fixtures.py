"""Relink the programs in fex_fixtures.py, and the C++ programs of the macOS fixtures when the tools
in fex_toolchain.py are there, and run them with aps5-fex.

    python3 run_fex_fixtures.py <relinker> <aps5-fex> <patched prx directory>

Each program reports its result through its exit status and standard output."""
import os
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fex_fixtures
import fex_toolchain

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


CHECKS = {"opening": created_with_mode}


def compiled_executable(directory, sources, libraries, extra=()):
    """Builds input.elf from C and C++ sources whose imports come from the named stub libraries, or
    from libc.prx when libraries is None."""
    objects = []
    imported, defined = set(), set()
    for source in sources:
        obj = directory / (pathlib.Path(source).stem + ".o")
        fex_toolchain.compile(MACOS / source, obj, extra=(["-x", "c"] if source.endswith(".c") else []) + list(extra))
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
    fex_toolchain.link_executable(renamed, stubs, directory / "input.elf")
    fex_toolchain.make_header_room(directory / "input.elf")


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


COMPILED = {
    "exception": (lambda d: compiled_executable(d, ["exception.cpp"], None), 43),
    "c-cleanup": (lambda d: compiled_executable(d, ["c_cleanup.c", "c_cleanup_main.cpp"], None), 47),
    "threads": (lambda d: compiled_executable(d, ["threads.cpp"], {"libc.prx": ["exit"],
                "libkernel.prx": ["scePthreadCreate", "scePthreadJoin"]}), 51),
    "jump": (lambda d: compiled_executable(d, [str(HERE / "jump.cpp")],
             {"libc.prx": ["setjmp", "longjmp", "exit"]}), 43),
    "fibers": (lambda d: compiled_executable(d, [str(HERE / "fibers.cpp")],
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
}


def main():
    relinker, runner, libraries = (pathlib.Path(argument).resolve() for argument in sys.argv[1:4])
    failures = []
    cases = {name: (lambda d, name=name: (d / "input.elf").write_bytes(fex_fixtures.FIXTURES[name]()), status, output)
             for name, (status, output) in EXPECTED.items()}
    if fex_toolchain.available():
        cases.update({name: (build, status, None) for name, (build, status) in COMPILED.items()})
    else:
        print("compiled programs skipped: clang, llvm-nm or an LLVM ELF linker is missing")
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
            (directory / "libs").symlink_to(libraries, target_is_directory=True)
            executed = subprocess.run([str(runner), "eboot.elf"], cwd=directory, capture_output=True, text=True, timeout=60)
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
