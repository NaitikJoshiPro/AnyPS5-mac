"""Parsers for tools/mac/profile_title.py (pure Python, no title, GPU or subprocess).

Inputs, one function each:
- `sample <pid>` call graphs: per-thread sample counts by category (guest JIT code, FEX, HLE libraries,
  AGC driver, Metal, waits, spins), with the hottest leaves.
- per-thread CPU counters by performance level (libproc PROC_PIDTHREADCOUNTS, via read_thread_counts):
  P-core and E-core time, cycles, instructions.
- `powermetrics --samplers cpu_power,gpu_power,thermal` text: cluster frequency and residency, GPU active
  frequency and residency, power, thermal pressure.
- `[gputime]` lines of APS5_PROFILE_GPU=1 and the guest sampler's lines (APS5_FEX_SAMPLE).
- the driver trace (aps5-trace-*.json.gz, via tools/trace_digest.py): per-thread phase time per frame and
  what each frame waited on.
"""
import collections
import ctypes
import re
import struct

# ---------------------------------------------------------------------------------------------------------
# sample(1) call graphs

_TREE_LINE = re.compile(r"^(?P<prefix>[ +!:|]*?)(?P<count>\d+) (?P<rest>\S.*)$")
_FRAME = re.compile(r"^(?P<symbol>.*?)\s+\(in (?P<module>[^)]*)\)")
_THREAD = re.compile(r"^Thread_(?P<tid>\d+|<multiple>)(?::\s*(?P<name>[^ ].*?))?(?:\s+DispatchQueue_\d+: (?P<queue>.*?)\s+\((?:serial|concurrent)\))?\s*$")

SYSTEM_MODULES = re.compile(
    r"^(libsystem_\w+\.dylib|libc\+\+(abi)?\.1?\.?dylib|libc\+\+\.1\.dylib|libc\+\+abi\.dylib|libdispatch\.dylib|"
    r"libobjc\.A\.dylib|libobjcMsgSend\d*\.dylib|dyld|libdyld\.dylib|libxpc\.dylib|CoreFoundation|Foundation|"
    r"libsystem_blocks\.dylib|libsystem_c\.dylib)$")
METAL_MODULES = re.compile(r"^(Metal|MetalTools|MetalFX|AGXMetal\w*|IOGPU|IOKit|IOSurface|QuartzCore|"
                           r"MetalPerformanceShaders|libGPUCompiler\w*|GPUCompiler|MTLCompiler\w*|CoreGraphics)$")
WAIT_LEAVES = {
    "__psynch_cvwait", "__psynch_mutexwait", "__psynch_rw_rdlock", "__psynch_rw_wrlock", "__semwait_signal",
    "__ulock_wait", "__ulock_wait2", "kevent", "kevent_id", "kevent_qos", "kevent64", "__workq_kernreturn",
    "semaphore_wait_trap", "semaphore_timedwait_trap", "mach_wait_until", "__select", "__sigsuspend",
    "__sigwait", "__pselect", "poll", "__wait4", "__read_nocancel", "read", "__recvfrom", "__accept",
}
FILE_IO_LEAVES = {"stat", "lstat", "fstat", "stat64", "lstat64", "fstat64", "open", "__open", "__open_nocancel",
                  "openat", "__openat", "close", "__close_nocancel", "pread", "__pread_nocancel", "__pread",
                  "pwrite", "__pwrite_nocancel", "getdirentries64", "__getdirentries64", "fsync", "fcntl",
                  "__fcntl", "getattrlist", "__getattrlist", "access", "__access_extended", "unlink", "rename"}
YIELD_LEAVES = {"swtch_pri", "thread_switch", "sched_yield", "swtch", "__swtch_pri"}
SLEEP_CALLERS = re.compile(r"(sceKernelUsleep|sceKernelNanosleep|usleep_nid|nanosleep_nid|SleepNanos|"
                           r"sceKernelSleep|scePthreadYield|sched_yield_nid)")

# FEX: anything under these frames is translation work (the JIT compiling a block), whatever the leaf.
FEX_COMPILE = re.compile(r"(CompileCode|CompileBlock|Frontend::Decoder|OpDispatchBuilder|Arm64JITCore::|"
                         r"IR::PassManager|RegisterAllocation|ConstProp|DeadCode|IREmitter|"
                         r"Core::ContextImpl::CompileCode|GenerateIR|LookupCache::AddBlock|"
                         r"CodeEmitter|DisassembleBlock|Arm64Emitter)")
FEX_SIGNAL = re.compile(r"(SignalDelegator|HandleSignal|HandleGuestSignal|SignalHandler|_sigtramp|"
                        r"GuestDisplaced\w*Fault|HandleSIGBUS|HandleSIGSEGV|BackpatchSite|WriteWatch\w*Fault|"
                        r"GuestWriteWatchHandleFault)")
FEX_THUNK = re.compile(r"(Aps5BridgeThunkCall|Bridge::Invoke|Aps5NativeCall|ThunkHandler|CallbackReturn|"
                       r"Aps5GuestCallback)")
FEX_DISPATCH = re.compile(r"(Dispatcher|LookupCache|ExitFunctionLink|BlockLink|ThreadManager|"
                          r"FEXCore::Context|HandleSyscall|SyscallHandler|CPUBackend)")
TSO_HINT = re.compile(r"(ParanoidTSO|HalfBarrier|TSO)")


def parse_frame(rest):
    """`symbol  (in module) + off  [addr] file:line` -> (symbol, module); a JIT frame is ("???", "<jit>")."""
    match = _FRAME.match(rest)
    if not match:
        return rest.strip(), ""
    symbol, module = match.group("symbol").strip(), match.group("module").strip()
    if module == "<unknown binary>":
        module = "<jit>"
    return symbol, module


def parse_sample(text):
    """Return {thread key: {"tid", "name", "queue", "total", "stacks": Counter(path tuple -> self count)}}.

    A path is a tuple of (symbol, module) from the thread's root to the leaf. Self counts are a node's count
    minus its children's, so every sample is counted once, at its leaf."""
    threads = {}
    in_graph = False
    stack = []  # (column, node)
    current = None
    for line in text.splitlines():
        if line.startswith("Call graph:"):
            in_graph = True
            continue
        if not in_graph:
            continue
        if not line.strip():
            if threads:
                break
            continue
        if line.startswith("Total number in stack") or line.startswith("Sort by top of stack"):
            break
        match = _TREE_LINE.match(line)
        if not match:
            continue
        column = len(match.group("prefix"))
        count = int(match.group("count"))
        rest = match.group("rest")
        thread = _THREAD.match(rest)
        if thread and column <= 4:
            while stack and current is not None:
                _close(stack.pop()[1], current)
            key = f"Thread_{thread.group('tid')}"
            if key in threads:
                key = f"{key}#{len(threads)}"
            current = {"tid": thread.group("tid"), "name": (thread.group("name") or "").strip(),
                       "queue": (thread.group("queue") or "").strip(), "total": count,
                       "stacks": collections.Counter()}
            threads[key] = current
            stack = [(column, {"frame": None, "count": count, "children": 0, "path": ()})]
            continue
        if current is None:
            continue
        while stack and stack[-1][0] >= column:
            _close(stack.pop()[1], current)
        parent = stack[-1][1] if stack else None
        frame = parse_frame(rest)
        node = {"frame": frame, "count": count, "children": 0,
                "path": (parent["path"] if parent else ()) + (frame,)}
        if parent is not None:
            parent["children"] += count
        stack.append((column, node))
    while stack and current is not None:
        _close(stack.pop()[1], current)
    return threads


def _close(node, thread):
    own = node["count"] - node["children"]
    if own > 0 and node["frame"] is not None:
        thread["stacks"][node["path"]] += own


def _first_owner(path):
    """The deepest frame that is not a system library: who asked for the system call."""
    for symbol, module in reversed(path):
        if module and not SYSTEM_MODULES.match(module) and module not in ("libsystem_kernel.dylib",) and \
                not symbol.startswith(("std::", "DYLD-STUB", "TimedWait::", "void std::")):
            return symbol, module
    return path[-1] if path else ("?", "?")


def short_symbol(symbol, width=90):
    """Drop the argument list, template arguments and ABI tags: `ns::(anonymous namespace)::f<T>(int) const`
    -> `ns::{anon}::f`."""
    symbol = symbol.replace("(anonymous namespace)", "{anon}")
    symbol = re.sub(r"\[abi:[^\]]*\]", "", symbol)
    if symbol.endswith(")") or symbol.endswith(") const"):
        depth = 0
        for index in range(len(symbol) - 1, -1, -1):
            depth += {")": 1, "(": -1}.get(symbol[index], 0)
            if depth == 0 and symbol[index] == "(":
                symbol = symbol[:index]
                break
    while True:
        stripped = re.sub(r"<[^<>]*>", "", symbol)
        if stripped == symbol:
            break
        symbol = stripped
    return symbol if len(symbol) <= width else symbol[: width - 3] + "..."


def classify_stack(path):
    """(category, detail) for one leaf path. Categories:
    guest JIT code; FEX compile; FEX signals; FEX thunk/bridge; FEX dispatcher; HLE <module>; AGC driver;
    Metal API; Metal shader compile; wait: <kind>; spin/yield; system; other."""
    if not path:
        return "other", "?"
    symbols = [symbol for symbol, _ in path]
    leaf_symbol, leaf_module = path[-1]
    joined = " ".join(symbols)
    if leaf_symbol in YIELD_LEAVES or leaf_symbol in ("cthread_yield", "pthread_yield_np"):
        owner = _first_owner(path)
        return "spin/yield", short_symbol(owner[0])
    blocking = leaf_symbol in WAIT_LEAVES or (
        leaf_module == "libsystem_kernel.dylib" and leaf_symbol.endswith("_trap") and
        re.search(r"(CFRunLoop|mach_msg_receive|dispatch_mach|_dispatch_sema|semaphore_wait|waitUntil|"
                  r"WaitEvent|clock_get_time|nanosleep)", joined))
    if blocking:
        if re.search(r"(waitUntilSignaledValue|WaitEvent)", joined):
            return "wait: GPU", short_symbol(_first_owner(path)[0])
        if SLEEP_CALLERS.search(joined):
            return "wait: guest sleep", short_symbol(_first_owner(path)[0])
        if re.search(r"(Recorder::Sync|waitUntilCompleted|WaitSerial|waitForEvent|MTLSharedEvent|SyncThrough)", joined):
            return "wait: GPU", short_symbol(_first_owner(path)[0])
        if leaf_symbol == "__psynch_mutexwait" or "lock" in leaf_symbol:
            return "wait: lock", short_symbol(_first_owner(path)[0])
        if leaf_symbol in ("__workq_kernreturn", "kevent_id", "kevent", "kevent_qos", "kevent64") and \
                len(path) <= 4:
            return "idle: workqueue", leaf_symbol
        return "wait: other", short_symbol(_first_owner(path)[0])
    if re.search(r"(waitUntilSignaledValue|waitUntilCompleted|waitUntilScheduled)", joined) and \
            (leaf_symbol.endswith("_trap") or leaf_symbol in WAIT_LEAVES):
        return "wait: GPU", short_symbol(_first_owner([frame for frame in path
                                                       if not METAL_MODULES.match(frame[1])])[0])
    if leaf_symbol in ("start_wqthread", "thread_start") and len(path) <= 2:
        return "idle: workqueue", leaf_symbol  # a parked or just-created pool thread with no frames of its own
    if leaf_module == "libsystem_kernel.dylib" and leaf_symbol in FILE_IO_LEAVES:
        return "kernel: file I/O", short_symbol(_first_owner(path)[0])
    if leaf_module == "<jit>":
        return "guest JIT code", "translated guest code"
    # Translation work is decided by the stack, not the leaf (the leaf is often malloc or memcpy).
    if any(module.startswith("aps5-fex") and FEX_COMPILE.search(symbol) for symbol, module in path):
        return "FEX compile", short_symbol(next(symbol for symbol, module in reversed(path)
                                                if module.startswith("aps5-fex")))
    if any(FEX_SIGNAL.search(symbol) for symbol in symbols[-12:]) or "_sigtramp" in symbols:
        return "FEX signals", short_symbol(_first_owner(path)[0])
    owner_symbol, owner_module = _first_owner(path)
    if METAL_MODULES.match(owner_module) or METAL_MODULES.match(leaf_module):
        if re.search(r"(newLibraryWithSource|newComputePipelineState|newRenderPipelineState|MTLCompiler|"
                     r"PipelineCompil|compileFunction|newFunction)", joined):
            return "Metal shader compile", short_symbol(owner_symbol)
        driver = next((symbol for symbol, module in reversed(path) if module == "libSceAgcDriver.prx"), None)
        return "Metal API", short_symbol(driver or owner_symbol)
    if owner_module == "libSceAgcDriver.prx":
        return "AGC driver", short_symbol(owner_symbol)
    if owner_module.endswith(".prx"):
        return f"HLE {owner_module[:-4]}", short_symbol(owner_symbol)
    if owner_module.startswith("aps5-fex"):
        if FEX_THUNK.search(owner_symbol):
            return "FEX thunk/bridge", short_symbol(owner_symbol)
        if FEX_DISPATCH.search(owner_symbol) or "FEXCore" in owner_symbol:
            return "FEX dispatcher", short_symbol(owner_symbol)
        return "FEX other", short_symbol(owner_symbol)
    if SYSTEM_MODULES.match(owner_module) or owner_module == "libsystem_kernel.dylib":
        return "system", short_symbol(owner_symbol)
    return "other", f"{short_symbol(owner_symbol)} ({owner_module})"


def thread_role(thread):
    """A readable role for a sampled thread, from its name, queue or the frames near its root."""
    name, queue = thread.get("name", ""), thread.get("queue", "")
    if name:
        return name
    roots = collections.Counter()
    for path, count in thread["stacks"].items():
        for symbol, module in path[:14]:
            roots[(symbol, module)] += count
    joined = " ".join(symbol for (symbol, _), _ in roots.most_common(60))
    for pattern, role in (
            (r"Aps5StartGuest|GuestCpu::Run", "guest main"),
            (r"QueueWorker|queueWorker|WorkerLoop|Driver::enqueue|DriverDetail::Driver::run", "queue worker"),
            (r"Presenter|presentLoop|PresentThread|presenter", "presenter"),
            (r"vblankLoop", "vblank"),
            (r"PipelineCache::worker|PipelineCompil|compileWorker|Prewarm", "pipeline compiler"),
            (r"PipelineCache::saver", "pipeline saver"),
            (r"ReleaseThreadMain", "recorder release"),
            (r"RunThread|StartNativeThread|Aps5GuestCallEntry|\?\?\?", "guest thread"),
            (r"AudioOut|Ngs2|SDL_RunAudio|AudioThread", "audio"),
            (r"DiskCache", "FEX disk cache"),
            (r"VideoOut|Vblank|vblank", "videoout"),
    ):
        if re.search(pattern, joined):
            return role
    if queue:
        return queue
    return "thread"


def summarize_sample(threads, top_leaves=8):
    """Per thread: total samples, category counts, the hottest details. Plus process totals."""
    result = {"threads": [], "totals": collections.Counter(), "details": collections.defaultdict(collections.Counter),
              "samples_per_thread": 0}
    for key, thread in threads.items():
        categories = collections.Counter()
        details = collections.Counter()
        for path, count in thread["stacks"].items():
            category, detail = classify_stack(path)
            categories[category] += count
            details[(category, detail)] += count
            result["totals"][category] += count
            result["details"][category][detail] += count
        result["samples_per_thread"] = max(result["samples_per_thread"], thread["total"])
        result["threads"].append({
            "key": key, "tid": thread["tid"], "role": thread_role(thread), "total": thread["total"],
            "categories": dict(categories.most_common()),
            "busy": sum(count for category, count in categories.items() if not category.startswith(("wait", "idle"))),
            "top": [(category, detail, count) for (category, detail), count in details.most_common(top_leaves)],
        })
    result["threads"].sort(key=lambda entry: -entry["busy"])
    return result


# ---------------------------------------------------------------------------------------------------------
# per-thread CPU counters by performance level (macOS 14+ libproc SPI PROC_PIDTHREADCOUNTS; level 0 is the
# performance cluster: sysctl hw.perflevel0.name)

PROC_PIDLISTTHREADIDS = 28
PROC_PIDTHREADID64INFO = 15
PROC_PIDTHREADCOUNTS = 34
_COUNTS_DATA = struct.Struct("<5Q")  # instructions, cycles, user mach time, system mach time, energy nJ


def decode_thread_counts(raw):
    """proc_threadcounts: u16 len, u16 reserved, u32 reserved, then `len` proc_threadcounts_data records."""
    if len(raw) < 8:
        return []
    count = struct.unpack_from("<H", raw, 0)[0]
    levels = []
    for index in range(count):
        offset = 8 + index * _COUNTS_DATA.size
        if offset + _COUNTS_DATA.size > len(raw):
            break
        instructions, cycles, user, system, energy = _COUNTS_DATA.unpack_from(raw, offset)
        levels.append({"instructions": instructions, "cycles": cycles, "time": user + system, "energy_nj": energy})
    return levels


def read_thread_counts(pid):
    """{thread id: {"name", "levels": [...]}} for every thread of `pid` (same user; no root needed)."""
    lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    ids = (ctypes.c_uint64 * 4096)()
    size = lib.proc_pidinfo(pid, PROC_PIDLISTTHREADIDS, ctypes.c_uint64(0), ids, ctypes.sizeof(ids))
    result = {}
    for index in range(max(size, 0) // 8):
        tid = ids[index]
        out = (ctypes.c_uint8 * 256)()
        got = lib.proc_pidinfo(pid, PROC_PIDTHREADCOUNTS, ctypes.c_uint64(tid), out, 256)
        info = (ctypes.c_uint8 * 512)()
        named = lib.proc_pidinfo(pid, PROC_PIDTHREADID64INFO, ctypes.c_uint64(tid), info, 512)
        name = bytes(info[named - 64:named]).split(b"\0")[0].decode("utf-8", "replace") if named >= 64 else ""
        if got > 0:
            result[tid] = {"name": name, "levels": decode_thread_counts(bytes(out[:got]))}
    return result


def mach_timebase_ns():
    class Timebase(ctypes.Structure):
        _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]
    info = Timebase()
    ctypes.CDLL("/usr/lib/libSystem.dylib").mach_timebase_info(ctypes.byref(info))
    return info.numer / info.denom


def diff_thread_counts(before, after, ns_per_tick, seconds):
    """Per-thread CPU over the window: ms on P and E cores, P share, IPC on each level. Threads that started
    inside the window count from zero; threads that ended are lost (their time is gone with them)."""
    rows = []
    for tid, now in after.items():
        then = before.get(tid, {"levels": [{"instructions": 0, "cycles": 0, "time": 0, "energy_nj": 0}] * len(now["levels"])})
        levels = []
        for index, level in enumerate(now["levels"]):
            old = then["levels"][index] if index < len(then["levels"]) else {"instructions": 0, "cycles": 0, "time": 0, "energy_nj": 0}
            levels.append({key: max(level[key] - old[key], 0) for key in level})
        p = levels[0] if levels else {"time": 0, "cycles": 0, "instructions": 0, "energy_nj": 0}
        e = levels[1] if len(levels) > 1 else {"time": 0, "cycles": 0, "instructions": 0, "energy_nj": 0}
        p_ms, e_ms = p["time"] * ns_per_tick / 1e6, e["time"] * ns_per_tick / 1e6
        total = p_ms + e_ms
        if total <= 0.0:
            continue
        rows.append({
            "tid": tid, "name": now.get("name", ""), "cpu_ms": total, "cpu_pct": 100.0 * total / (seconds * 1000.0),
            "p_ms": p_ms, "e_ms": e_ms, "p_share": p_ms / total,
            "p_ipc": p["instructions"] / p["cycles"] if p["cycles"] else 0.0,
            "e_ipc": e["instructions"] / e["cycles"] if e["cycles"] else 0.0,
            "p_ghz": p["cycles"] / (p_ms * 1e6) if p_ms > 0 else 0.0,
            "e_ghz": e["cycles"] / (e_ms * 1e6) if e_ms > 0 else 0.0,
            "energy_mj": (p["energy_nj"] + e["energy_nj"]) / 1e6,
        })
    rows.sort(key=lambda row: -row["cpu_ms"])
    return rows


# ---------------------------------------------------------------------------------------------------------
# powermetrics

_PM_PATTERNS = {
    "e_mhz": re.compile(r"^E-Cluster HW active frequency:\s+(\d+) MHz"),
    "e_active_pct": re.compile(r"^E-Cluster HW active residency:\s+([\d.]+)%"),
    "p_mhz": re.compile(r"^P-Cluster HW active frequency:\s+(\d+) MHz"),
    "p_active_pct": re.compile(r"^P-Cluster HW active residency:\s+([\d.]+)%"),
    "gpu_mhz": re.compile(r"^GPU HW active frequency:\s+(\d+) MHz"),
    "gpu_active_pct": re.compile(r"^GPU HW active residency:\s+([\d.]+)%"),
    "cpu_mw": re.compile(r"^CPU Power:\s+(\d+) mW"),
    "gpu_mw": re.compile(r"^GPU Power:\s+(\d+) mW"),
}
_PM_CPU = re.compile(r"^CPU (\d+) active residency:\s+([\d.]+)%")
_PM_PRESSURE = re.compile(r"^Current pressure level:\s+(\w+)")
_PM_MAX_MHZ = re.compile(r"(\d+) MHz:\s+([\d.]+)%")


def parse_powermetrics(text):
    """One dict per `*** Sampled system activity` block, plus a summary with medians and the pressure levels."""
    samples = []
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("*** Sampled system activity"):
            current = {"cpu_active_pct": {}}
            samples.append(current)
            continue
        if current is None:
            continue
        for key, pattern in _PM_PATTERNS.items():
            match = pattern.match(line)
            if match and key not in current:
                current[key] = float(match.group(1))
                if key == "gpu_active_pct":
                    # The highest GPU state with residency: is the GPU capped below its top frequency?
                    states = [(int(mhz), float(pct)) for mhz, pct in _PM_MAX_MHZ.findall(line)]
                    current["gpu_states"] = states
        match = _PM_CPU.match(line)
        if match:
            core = int(match.group(1))
            current["cpu_active_pct"][core] = float(match.group(2))
            states = [(int(mhz), float(pct)) for mhz, pct in _PM_MAX_MHZ.findall(line)]
            current.setdefault("cpu_states", {})[core] = states
        match = _PM_PRESSURE.match(line)
        if match:
            current["pressure"] = match.group(1)
    return samples


def _median(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0


def summarize_powermetrics(samples, p_cores=(4, 5, 6, 7), e_cores=(0, 1, 2, 3)):
    if not samples:
        return {}
    summary = {key: _median([sample.get(key) for sample in samples]) for key in _PM_PATTERNS}
    summary["samples"] = len(samples)
    summary["pressure"] = dict(collections.Counter(sample.get("pressure", "?") for sample in samples))
    for label, cores in (("p", p_cores), ("e", e_cores)):
        per_core = [sum(sample["cpu_active_pct"].get(core, 0.0) for core in cores) / len(cores) for sample in samples]
        summary[f"{label}_core_active_pct"] = _median(per_core)
        # The cluster's "HW active frequency" line often reads 0 MHz on macOS 27 while its cores ran; the
        # per-core residency histograms give the frequency the cluster actually ran at when active.
        weighted = active = 0.0
        for sample in samples:
            for core in cores:
                for mhz, pct in sample.get("cpu_states", {}).get(core, []):
                    weighted += mhz * pct
                    active += pct
        summary[f"{label}_active_mhz"] = weighted / active if active else None
    peak = 0
    for sample in samples:
        for mhz, pct in sample.get("gpu_states", []):
            if pct > 0.0:
                peak = max(peak, mhz)
    summary["gpu_peak_state_mhz"] = peak
    return summary


# ---------------------------------------------------------------------------------------------------------
# [gputime] (APS5_PROFILE_GPU=1) and the guest sampler

_GPUTIME = re.compile(
    r"\[gputime\] (?P<program_ms>\d+) ms of GPU time in (?P<batches>\d+) batches over (?P<seconds>\d+) s "
    r"\(batch (?P<batch_ms>\d+) ms first to last command; classes (?P<classes_ms>\d+) ms, all ranges "
    r"(?P<ranges_ms>\d+) ms, untimed (?P<untimed_ms>\d+) ms outside every range; (?P<presents>\d+) presents; "
    r"(?P<dropped>\d+) ranges dropped at the \d+ cap(?:, (?P<invalid>\d+) with an unwritten or reversed stamp)?\)")
_GPUTIME_CLASS = re.compile(r"([\w-]+) x(\d+) ([\d.]+)ms ([\d.]+)MiB")


def parse_gputime(text):
    """Every `[gputime]` line (APS5_PROFILE_GPU=1, Recorder.mm): the batch totals and the per-class time."""
    lines = []
    for line in text.splitlines():
        match = _GPUTIME.search(line)
        if not match:
            continue
        entry = {key: int(value or 0) for key, value in match.groupdict().items()}
        classes = {}
        if "by class:" in line:
            for name, count, ms, mib in _GPUTIME_CLASS.findall(line.split("by class:", 1)[1]):
                classes[name] = {"count": int(count), "ms": float(ms), "mib": float(mib)}
        entry["classes"] = classes
        lines.append(entry)
    return lines


_FEX_WINDOW = re.compile(r"\[fex-sample\] window (\d+): ([\d.]+) s, (\d+) guest threads")
_FEX_THREAD = re.compile(r'\[fex-sample\] thread (\d+) "(.*)": (\d+) samples, (\d+) distinct RIPs')
_FEX_RIP = re.compile(r"\[fex-sample\]\s+([\d.]+)% rip 0x([0-9a-f]+) (\S+)\+0x([0-9a-f]+) (jit|host)\s*(.*)$")


def parse_fex_sample(text):
    """Guest sampler reports (APS5_FEX_SAMPLE, GuestSampler.cpp): per window, per guest thread, the sample
    count and its top RIPs, each `jit` (translated code) or `host` (native: a bridge into our libraries, a
    system call, a signal handler)."""
    windows = []
    window = thread = None
    for line in text.splitlines():
        match = _FEX_WINDOW.search(line)
        if match:
            window = {"index": int(match.group(1)), "seconds": float(match.group(2)),
                      "registered": int(match.group(3)), "threads": []}
            windows.append(window)
            thread = None
            continue
        match = _FEX_THREAD.search(line)
        if match and window is not None:
            thread = {"thread": int(match.group(1)), "name": match.group(2), "samples": int(match.group(3)),
                      "rips": []}
            window["threads"].append(thread)
            continue
        match = _FEX_RIP.search(line)
        if match and thread is not None:
            thread["rips"].append({"pct": float(match.group(1)), "rip": int(match.group(2), 16),
                                   "module": match.group(3), "offset": int(match.group(4), 16),
                                   "kind": match.group(5), "host": match.group(6).strip()})
    return windows


def summarize_fex_sample(windows):
    """Over every window: guest samples per thread and, of the top RIPs, the share in translated code,
    in native code that waits (system call or wait) and in other native code."""
    totals = collections.Counter()
    kinds = collections.Counter()
    hot = collections.Counter()
    for window in windows:
        for thread in window["threads"]:
            totals[thread["thread"]] += thread["samples"]
            for rip in thread["rips"]:
                samples = rip["pct"] * thread["samples"] / 100.0
                if rip["kind"] == "jit":
                    kind = "jit"
                elif re.search(r"(__psynch|__semwait|__read|__ulock|kevent|mach_msg|nanosleep|__workq|"
                               r"semaphore|__select|cvwait|_sigtramp)", rip["host"]):
                    kind = "native wait/syscall/signal"
                else:
                    kind = "native work"
                kinds[kind] += samples
                hot[(rip["module"], rip["offset"], rip["kind"], rip["host"].split("+0x")[0])] += samples
    return {"samples": sum(totals.values()), "threads": len(totals), "kinds": dict(kinds),
            "hot": [(module, offset, kind, host, samples) for (module, offset, kind, host), samples in hot.most_common(12)]}


# ---------------------------------------------------------------------------------------------------------
# the driver trace: what each frame waits on

def frame_critical_path(trace, wait_hint=re.compile(r"(wait|Sync|WaitSerial|GpuMutex)", re.I)):
    """Per thread, the mean ms per frame in each top-level phase and in each wait phase at any depth, over the
    frames of a trace_digest.Trace. The thread whose busy time per frame is closest to the frame time is the
    critical path; its largest wait names what the frame waits on."""
    frames = trace.frames(None)
    if not frames:
        return {"frames": 0, "threads": []}
    start, end = frames[0][1], frames[-1][2]
    count = len(frames)
    frame_ms = (end - start) / 1000.0 / count
    threads = []
    for tid, slices in trace.slices.items():
        top = collections.Counter()
        waits = collections.Counter()
        for ts, dur, name, depth, under_wait in slices:
            if ts < start or ts >= end:
                continue
            if depth == 0:
                top[name] += dur
            if wait_hint.search(name.rsplit(".", 1)[-1]) and not under_wait:
                waits[name] += dur
        if not top:
            continue
        busy = sum(top.values()) / 1000.0 / count
        wait = sum(waits.values()) / 1000.0 / count
        threads.append({
            "thread": trace.name(tid), "busy_ms": busy, "wait_ms": wait, "work_ms": max(busy - wait, 0.0),
            "top": [(name, value / 1000.0 / count) for name, value in top.most_common(6)],
            "waits": [(name, value / 1000.0 / count) for name, value in waits.most_common(6)],
        })
    threads.sort(key=lambda entry: -entry["busy_ms"])
    return {"frames": count, "frame_ms": frame_ms, "threads": threads}


_FEX_STATS = re.compile(r"\[fex-stats\] window (\d+): ([\d.]+) s, (\d+) guest threads; JIT (\d+) blocks in ([\d.]+) ms "
                        r"\(([\d.]+) us each\), lookup-cache misses (\d+), cache lock read ([\d.]+) ms write ([\d.]+) ms, "
                        r"disk cache (\d+) hits (\d+) misses \(([\d.]+) ms lookup\), SMC (\d+)")


def parse_fex_stats(text):
    """`[fex-stats]` lines (APS5_FEX_STATS, core/fex/src/GuestStats.cpp), one dict per window."""
    keys = ("window", "seconds", "threads", "jit_blocks", "jit_ms", "jit_us_each", "cache_misses", "read_lock_ms",
            "write_lock_ms", "disk_hits", "disk_misses", "disk_lookup_ms", "smc")
    rows = []
    for match in _FEX_STATS.finditer(text):
        rows.append({key: (float(value) if "." in value else int(value)) for key, value in zip(keys, match.groups())})
    return rows
