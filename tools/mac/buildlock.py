#!/usr/bin/env python3
"""Runs a command while holding an exclusive lock on a build directory.

Several agents can share one build directory (the port keeps at most two), but
two ninja processes in the same directory corrupt its logs. Wrap every build or
test command that touches a shared build directory:

    tools/mac/buildlock.py build-arm64 -- ninja -C build-arm64 libs
    tools/mac/buildlock.py build-arm64 -- ctest --test-dir build-arm64 -R guest_tcp

The lock is released when the command exits, including on a crash.
"""
import fcntl
import os
import subprocess
import sys
import time


def main() -> int:
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        sys.stderr.write("usage: buildlock.py <build-dir> -- <command...>\n")
        return 2
    build_dir = sys.argv[1]
    command = sys.argv[3:]
    os.makedirs(build_dir, exist_ok=True)
    lock_path = os.path.join(build_dir, ".aps5-build.lock")
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
