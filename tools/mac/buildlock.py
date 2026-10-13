#!/usr/bin/env python3
"""Runs a command while holding an exclusive lock on a build directory.

Several agents can share one build directory (the port keeps at most two), but
two ninja processes in the same directory corrupt its logs. Wrap every build or
test command that touches a shared build directory:

    tools/mac/buildlock.py build-arm64 -- ninja -C build-arm64 libs
    tools/mac/buildlock.py build-arm64 -- ctest --test-dir build-arm64 -R guest_tcp

The lock is released when the command exits, including on a crash.

A gpu_suite run freezes the build directory for its whole duration with a suite lease
(<build-dir>/.aps5-suite-lease.json: pid, start time, max duration, command line). While a lease exists whose pid is
alive and whose age is under its max, buildlock waits (with a periodic message) before taking the lock. A stale
lease (dead pid, or over its max) is ignored and removed. To see who holds it:

    cat <build-dir>/.aps5-suite-lease.json
"""
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


def main() -> int:
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        sys.stderr.write("usage: buildlock.py <build-dir> -- <command...>\n")
        return 2
    build_dir = sys.argv[1]
    command = sys.argv[3:]
    os.makedirs(build_dir, exist_ok=True)
    owner = os.environ.get("APS5_SUITE_LEASE_OWNER")
    owner_pid = int(owner) if owner and owner.isdigit() else None
    lock_path = os.path.join(build_dir, ".aps5-build.lock")
    # Wait for a suite lease without holding the lock, so the suite's own tests can still take it.
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
