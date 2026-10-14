#!/usr/bin/env python3
"""Runs a command while holding an exclusive lock on a build directory.

Several agents can share one build directory (the port keeps at most two), but
two ninja processes in the same directory corrupt its logs. Wrap every build or
test command that touches a shared build directory:

    tools/mac/buildlock.py build-arm64 -- ninja -C build-arm64 libs
    tools/mac/buildlock.py build-arm64 -- ctest --test-dir build-arm64 -R guest_tcp

The build lock is per build directory. GPU work must also take the machine-wide GPU lock (one flock shared by every
worktree, <repo>/.mac-work/locks/gpu.lock, holder recorded in gpu.lock.info); gpu_suite and title_run do it themselves,
for ad-hoc commands pass --gpu:  tools/mac/buildlock.py --gpu build-arm64 -- <command>

The lock is released when the command exits, including on a crash.

A gpu_suite run freezes the build directory for its whole duration with a suite lease
(<build-dir>/.aps5-suite-lease.json: pid, start time, max duration, command line). While a lease exists whose pid is
alive and whose age is under its max, buildlock waits (with a periodic message) before taking the lock. A stale
lease (dead pid, or over its max) is ignored and removed. To see who holds it:

    cat <build-dir>/.aps5-suite-lease.json
"""
import contextlib
import fcntl
import json
import os
import subprocess
import sys
import time


LEASE_NAME = ".aps5-suite-lease.json"
LEASE_POLL = 5.0
LEASE_MESSAGE_EVERY = 60.0


def lease_path(build_dir) -> str:
    return os.path.join(str(build_dir), LEASE_NAME)


def read_lease(build_dir, now=None):
    """The live lease of a build directory, or None. A stale or unreadable lease is removed."""
    path = lease_path(build_dir)
    try:
        with open(path, encoding="utf-8") as stream:
            lease = json.load(stream)
        pid, started, limit = int(lease["pid"]), float(lease["started"]), float(lease["max_seconds"])
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError):
        pid = started = limit = None
    if pid is not None:
        alive = True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            alive = False
        except PermissionError:
            pass
        if alive and (now if now is not None else time.time()) - started < limit:
            return lease
    try:
        os.unlink(path)
    except OSError:
        pass
    return None


def describe_lease(lease) -> str:
    started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(lease["started"])))
    return f"gpu_suite run {lease.get('run', '?')} (pid {lease['pid']}, started at {started}, max {float(lease['max_seconds']) / 60:.0f} min)"


def wait_for_lease(build_dir, owner_pid=None, poll=LEASE_POLL, message_every=LEASE_MESSAGE_EVERY, out=None) -> float:
    """Blocks while another process's live suite lease exists; returns the seconds waited."""
    out = out or sys.stderr
    start = last = time.monotonic()
    first = True
    while True:
        lease = read_lease(build_dir)
        if lease is None or (owner_pid is not None and int(lease["pid"]) == owner_pid):
            return time.monotonic() - start
        if first or time.monotonic() - last >= message_every:
            out.write(f"buildlock: waiting for {describe_lease(lease)}\n")
            out.flush()
            first, last = False, time.monotonic()
        time.sleep(poll)


def take_lease(build_dir, max_seconds, run="", poll=LEASE_POLL, out=None):
    """Creates the lease (waiting for another live one first) and returns its path."""
    os.makedirs(str(build_dir), exist_ok=True)
    path = lease_path(build_dir)
    record = {"pid": os.getpid(), "started": time.time(), "max_seconds": float(max_seconds), "run": run,
              "command": " ".join(sys.argv)}
    while True:
        wait_for_lease(build_dir, poll=poll, out=out)
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream)
        return path


def release_lease(build_dir) -> None:
    """Removes the lease if this process owns it."""
    path = lease_path(build_dir)
    try:
        with open(path, encoding="utf-8") as stream:
            owner = int(json.load(stream)["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return
    if owner == os.getpid():
        try:
            os.unlink(path)
        except OSError:
            pass


# ---- machine-wide GPU lock ---------------------------------------------------------------------------------
# One flock on a fixed path outside every worktree (<repo>/.mac-work/locks/gpu.lock, shared by all worktrees), so GPU
# workloads from different build directories never overlap. A sidecar file (gpu.lock.info) says who holds it. The
# kernel drops the flock when the holder dies, so a dead holder never blocks; a stale sidecar is just overwritten.
# Global lock order: suite lease (wait only, never held while blocking on others) -> GPU lock -> per-build flock.
# gpu_suite holds its lease for the whole run and takes the GPU lock per test, so anything that takes the GPU lock
# must first wait for the build's lease WITHOUT holding the GPU lock (wait_for_lease), and must not wait for the
# lease again once it holds the GPU lock (a lease holder may be queued on the GPU lock; waiting would deadlock).
# That skip is safe for run-only commands: while we hold the GPU lock the lease holder cannot run a test, so
# the binaries are stable unless this command itself rebuilds them (then the suite notices changed mtimes and fails).

GPU_LOCK_ENV = "APS5_GPU_LOCK"            # overrides the lock path (tests)
GPU_HELD_ENV = "APS5_GPU_LOCK_HELD"       # set in the environment of children of a holder: they must not re-take it
GPU_MESSAGE_EVERY = 60.0


def gpu_lock_path() -> str:
    override = os.environ.get(GPU_LOCK_ENV)
    if override:
        return override
    here = os.path.dirname(os.path.abspath(__file__))
    probe = here
    while True:  # a worktree lives at <repo>/.mac-work/wt/<name>: the enclosing .mac-work is the shared one
        if os.path.basename(probe) == ".mac-work":
            return os.path.join(probe, "locks", "gpu.lock")
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        common = subprocess.run(["git", "-C", here, "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                capture_output=True, text=True, check=True).stdout.strip()
        return os.path.join(os.path.dirname(common), ".mac-work", "locks", "gpu.lock")
    except (OSError, subprocess.CalledProcessError):
        return os.path.join(os.path.dirname(os.path.dirname(here)), ".mac-work", "locks", "gpu.lock")


def read_gpu_holder(path=None):
    """The sidecar record of the last holder, or None."""
    try:
        with open((path or gpu_lock_path()) + ".info", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return None


def describe_gpu_holder(info) -> str:
    if not info:
        return "an unknown holder"
    try:
        started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(info["started"])))
    except (KeyError, TypeError, ValueError):
        started = "?"
    return f"pid {info.get('pid', '?')} running `{info.get('command', '?')}` since {started}"


@contextlib.contextmanager
def gpu_lock(label="", out=None, poll=1.0, message_every=GPU_MESSAGE_EVERY, path=None):
    """Holds the machine-wide GPU lock for the with block, printing who holds it while waiting."""
    if os.environ.get(GPU_HELD_ENV):
        yield
        return
    out = out or sys.stderr
    path = path or gpu_lock_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    start = time.monotonic()
    with open(path, "a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            last = None
            while True:
                if last is None or time.monotonic() - last >= message_every:
                    out.write(f"gpu lock: waiting for {path}, held by {describe_gpu_holder(read_gpu_holder(path))}\n")
                    out.flush()
                    last = time.monotonic()
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(poll)
            out.write(f"gpu lock: acquired after {time.monotonic() - start:.0f} s\n")
            out.flush()
        info = {"pid": os.getpid(), "started": time.time(), "command": label or " ".join(sys.argv)}
        tmp = f"{path}.info.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as stream:
            json.dump(info, stream)
        os.replace(tmp, path + ".info")
        previous = os.environ.get(GPU_HELD_ENV)
        os.environ[GPU_HELD_ENV] = str(os.getpid())
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(GPU_HELD_ENV, None)
            else:
                os.environ[GPU_HELD_ENV] = previous
            try:
                if (read_gpu_holder(path) or {}).get("pid") == os.getpid():
                    os.unlink(path + ".info")
            except OSError:
                pass


def main() -> int:
    argv = sys.argv[1:]
    use_gpu = bool(argv) and argv[0] == "--gpu"
    if use_gpu:
        sys.argv.pop(1)
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        sys.stderr.write("usage: buildlock.py [--gpu] <build-dir> -- <command...>\n"
                         "  --gpu also takes the machine-wide GPU lock (for ad-hoc GPU commands)\n")
        return 2
    if use_gpu:
        wait_for_lease(sys.argv[1], int(os.environ["APS5_SUITE_LEASE_OWNER"]) if os.environ.get("APS5_SUITE_LEASE_OWNER", "").isdigit() else None)
        with gpu_lock(label=" ".join(sys.argv[3:])):
            return build_locked()
    return build_locked()


def build_locked() -> int:
    build_dir = sys.argv[1]
    command = sys.argv[3:]
    os.makedirs(build_dir, exist_ok=True)
    owner = os.environ.get("APS5_SUITE_LEASE_OWNER")
    owner_pid = int(owner) if owner and owner.isdigit() else None
    lock_path = os.path.join(build_dir, ".aps5-build.lock")
    # Wait for a suite lease without holding the lock, so the suite's own tests can still take it.
    # Under the GPU lock the lease was already waited for before taking it (see the lock order above): do not wait again.
    if not os.environ.get(GPU_HELD_ENV):
        wait_for_lease(build_dir, owner_pid)
    with open(lock_path, "a+") as lock:
        start = time.monotonic()
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.stderr.write(f"buildlock: waiting for {lock_path}\n")
            fcntl.flock(lock, fcntl.LOCK_EX)
            sys.stderr.write(f"buildlock: acquired after {time.monotonic() - start:.0f} s\n")
        return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
