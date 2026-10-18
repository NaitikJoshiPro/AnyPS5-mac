import argparse
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

DEVICE_LIMITATION = re.compile(r"skipped,.*(subgroup|the device (has no|reports|lacks|does not|imports no|flushes|binds fewer)|no display|no usable Vulkan device|host imports are unavailable)", re.IGNORECASE)

NO_DISPLAY = re.compile(r"skipped,.*no display", re.IGNORECASE)

# What ctest itself writes into <skipped message=...>; the test's own "skipped, <reason>" line says more.
CTEST_MESSAGE = re.compile(r"skipped, (?:SKIP_RETURN_CODE=\d+|SKIP_REGULAR_EXPRESSION_MATCHED|Disabled)$")

HIGH_REFRESH_LIMITATION = re.compile(r"current display does not provide 120 Hz", re.IGNORECASE)

WINDOWS_PREFIX = "agc"

# macOS (Metal): a skip is expected only when tools/mac/m2_skip_allowlist.txt lists the test with the reason it gives and
# the docs/dev/TechnicalDebt.md anchor that documents the Apple GPU or Metal limit (docs/mac/M2_DESIGN.md, D-M2-8).
REPOSITORY = Path(__file__).resolve().parents[1]
MACOS_ALLOWLIST = REPOSITORY / "tools" / "mac" / "m2_skip_allowlist.txt"
TECHNICAL_DEBT = REPOSITORY / "docs" / "dev" / "TechnicalDebt.md"
NOT_BUILT = re.compile(r"Unable to find executable")
ANCHOR_ID = re.compile(r"""\bid=["']([^"']+)["']""")
HEADING = re.compile(r"^#+\s+(.+?)\s*$", re.M)
SKIP_INSIDE = re.compile(r"\bskipped, .*")


@dataclass(frozen=True)
class Allowed:
    test: str
    reason: re.Pattern
    anchor: str
    line: int


def allowlist(path):
    """{test: Allowed} from lines '<test> | <reason regex> | <TechnicalDebt.md anchor>'; '#' starts a comment."""
    entries = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) != 3 or not all(fields):
            raise ValueError(f"{path}:{number}: expected '<test> | <reason regex> | <TechnicalDebt.md anchor>'")
        test, pattern, anchor = fields
        if test in entries:
            raise ValueError(f"{path}:{number}: {test} is listed twice")
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as error:
            raise ValueError(f"{path}:{number}: bad reason pattern: {error}") from error
        entries[test] = Allowed(test, compiled, anchor.lstrip("#"), number)
    return entries


def anchors(markdown):
    """Explicit ids (<a id="...">) and the GitHub slugs of the headings of a markdown text."""
    found = set(ANCHOR_ID.findall(markdown))
    for heading in HEADING.findall(markdown):
        found.add(re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-"))
    return found


def macos_reason(output):
    """The test's own skip line: one that starts with "skipped", else a "skipped, ..." inside a line ("<case>: skipped, ..."), else ctest's message."""
    lines = [line.strip() for line in output.splitlines()]
    own = [line for line in lines if line.startswith("skipped") and not CTEST_MESSAGE.match(line)]
    if own:
        return own[0]
    for line in lines:
        found = SKIP_INSIDE.search(line)
        if found and not CTEST_MESSAGE.match(found.group(0)):
            return found.group(0)
    return reason(output)


def statuses(path):
    """{test: ctest status} for every testcase of a JUnit file (run, fail, notrun, disabled)."""
    return {case.get("name"): case.get("status", "") for case in ET.parse(path).getroot().iter("testcase") if case.get("name")}


def macos_check(found, ran, entries, markdown):
    """(errors, warnings) for the skips of a macOS job: found = [(test, output)], ran = {test: status}."""
    errors, warnings = [], []
    known = anchors(markdown)
    for entry in entries.values():
        if entry.anchor not in known:
            errors.append(f"allowlist line {entry.line}: {entry.test}: TechnicalDebt.md has no anchor #{entry.anchor}")
    skipped = set()
    for name, output in found:
        skipped.add(name)
        why = macos_reason(output)
        if NOT_BUILT.search(output):
            errors.append(f"{name} was not built ({why})")
            continue
        entry = entries.get(name)
        if entry is None:
            errors.append(f"{name} skipped and is not in the allowlist: {why}")
        elif not entry.reason.search(why):
            errors.append(f"{name} skipped for another reason than the allowlist expects ({entry.reason.pattern!r}): {why}")
    for entry in entries.values():
        if entry.test in ran and entry.test not in skipped:
            warnings.append(f"stale allowlist entry (line {entry.line}): {entry.test} ran ({ran[entry.test]}) instead of skipping")
    return errors, warnings


def reason(output):
    lines = [line.strip() for line in output.splitlines() if line.strip().lower().startswith("skipped")]
    return next((line for line in lines if not CTEST_MESSAGE.match(line)), lines[0] if lines else "")


def high_refresh_limitation(output):
    reasons = []
    for line in output.splitlines():
        reason = line.strip()
        if reason.lower().startswith("skipped,"):
            reason = reason[len("skipped,"):].strip()
        if reason:
            reasons.append(reason)
    return bool(reasons) and all(HIGH_REFRESH_LIMITATION.fullmatch(reason) for reason in reasons)


def skips(path):
    for case in ET.parse(path).getroot().iter("testcase"):
        if case.get("status") == "disabled":
            raise ValueError(f"{path}: testcase {case.get('name')!r} is disabled and was not run")
        skipped = case.find("skipped")
        if skipped is None:
            continue
        name = case.get("name")
        if not name:
            raise ValueError(f"{path}: skipped testcase has no name")
        output = ["skipped, " + reason.strip() for reason in (skipped.get("message", ""), skipped.text or "") if reason.strip()]
        output += [case.findtext(channel) or "" for channel in ("system-out", "system-err")]
        yield name, "\n".join(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fail when a test is skipped that the job's device is not expected to skip")
    parser.add_argument("junit", type=Path, help="the file written by ctest --output-junit")
    parser.add_argument("--job", choices=("linux", "windows", "macos"), default="linux")
    parser.add_argument("--list", action="store_true", help="print the skipped tests and why the file says they skipped")
    parser.add_argument("--allowlist", type=Path, default=MACOS_ALLOWLIST, help="macos: the skips the job expects (default: tools/mac/m2_skip_allowlist.txt)")
    parser.add_argument("--technical-debt", type=Path, default=TECHNICAL_DEBT, help="macos: where the allowlist anchors live (default: docs/dev/TechnicalDebt.md)")
    args = parser.parse_args()
    try:
        found = sorted(skips(args.junit))
    except (OSError, ET.ParseError, ValueError) as error:
        parser.error(str(error))
    if args.list:
        for name, output in found:
            print(f"{name}: {reason(output)}")
        raise SystemExit
    if args.job == "macos":
        try:
            errors, warnings = macos_check(found, statuses(args.junit), allowlist(args.allowlist), args.technical_debt.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            parser.error(str(error))
        for warning in warnings:
            print(f"warning: {warning}")
        for error in errors:
            print(f"error: {error}")
        if errors:
            print(f"{len(errors)} problem(s): a macOS skip needs an allowlist entry with its reason and a TechnicalDebt.md anchor ({args.allowlist})")
            raise SystemExit(1)
        print(f"{len(found)} test(s) skipped, all allowlisted ({args.allowlist})")
        raise SystemExit
    if args.job == "windows":
        unexpected = [name for name, output in found if not name.startswith(WINDOWS_PREFIX) and not NO_DISPLAY.search(output) and not high_refresh_limitation(output)]
        expected = f"any {WINDOWS_PREFIX}* test, which the runner has no Vulkan driver for, or a test that reports no display or Vulkan device or a display without a 120 Hz mode"
    else:
        unexpected = [name for name, output in found if not DEVICE_LIMITATION.search(output) and not high_refresh_limitation(output)]
        expected = "any test that reports the device limitation it needs, such as narrow subgroups, a missing extension, no display for a Vulkan window or a display without a 120 Hz mode"
    for name in unexpected:
        print(f"error: {name} skipped")
    if unexpected:
        print(f"{len(unexpected)} test(s) skipped for a reason that is not expected: the job lost that coverage and ctest still reports every test as passed")
        raise SystemExit(1)
    print(f"{len(found)} test(s) skipped, all expected ({expected})")
