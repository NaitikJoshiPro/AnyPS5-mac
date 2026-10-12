"""Runs a host test program, then guest fixtures under aps5-fex: for the libkernel tests whose subject
(the guest's x86-64 register state) lives in the FEXCore runner on arm64. The host program checks the
arguments and exits 0 or 77 (it cannot reach a guest's registers); the fixtures run the operations.

    python3 run_host_and_fixtures.py <host program> [arguments...] -- <relinker> <aps5-fex> <prx directory> <fixture>...
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    separator = sys.argv.index("--")
    host, fixture = sys.argv[1:separator], sys.argv[separator + 1:]
    done = subprocess.run(host, timeout=60)
    if done.returncode not in (0, 77):
        print(f"{' '.join(host)} exited {done.returncode}")
        return 1
    relinker, runner, libraries, *names = fixture
    return subprocess.run([sys.executable, os.path.join(HERE, "run_fex_fixtures.py"), relinker, runner, libraries, "--only", *names],
                          timeout=300).returncode


if __name__ == "__main__":
    sys.exit(main())
