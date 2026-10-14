#!/usr/bin/env python3
"""Progress numbers of the native Apple Silicon port, as badges and a page.

    mac_progress.py site OUT [--no-upstream]   write the Pages site into OUT
    mac_progress.py gpu-snapshot [--run DIR]  copy the latest full gpu_suite run into docs/mac/gpu-status.json
    mac_progress.py show                      print the Mac metrics

The site holds upstream's two metrics, made by tools/progress.py and tools/progress_history.py unchanged
(system libraries, shader instructions, history), plus three Mac-port metrics:

    port milestones  verified exit criteria from docs/mac/milestones.toml (MISSION §8), each milestone weighted
                     equally, each criterion equally inside its milestone; "partial" counts as 0
    Metal GPU tests  passed / registered GPU tests of the last full gpu_suite run, from the committed snapshot
                     docs/mac/gpu-status.json (CI has no Metal device); allowlisted skips and held tests are
                     shown separately and count as not passed
    titles           the stage of every title in the Titles table of docs/mac/COMPATIBILITY.md on a 0-100 scale
                     (Nothing 0, Loads/Boots 20, Intro 40, Menu 60, In game 80, Playable 100), averaged

Everything is computed from files in the tree, so CI and a local run give the same numbers.
"""
import argparse
import html
import importlib.util
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MILESTONES = Path("docs/mac/milestones.toml")
PROGRESS_LOG = Path("docs/mac/PROGRESS.md")
COMPATIBILITY = Path("docs/mac/COMPATIBILITY.md")
GPU_STATUS = Path("docs/mac/gpu-status.json")
GPU_LOGS = Path(".mac-work/logs/gpu-suite")
BRANCH = "mac-native"
STATUSES = ("verified", "partial", "open")

# Longest names first so that "In game" is not read as something shorter.
STAGES = (("playable", 100, "Playable"), ("in game", 80, "In game"), ("in-game", 80, "In game"),
          ("ingame", 80, "In game"), ("menu", 60, "Menu"), ("intro", 40, "Intro"), ("loads", 20, "Loads"),
          ("boots", 20, "Loads"), ("nothing", 0, "Nothing"))


def load_upstream_progress():
    spec = importlib.util.spec_from_file_location("anyps5_progress", REPO / "tools" / "progress.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percent(numerator, denominator):
    return round(100 * numerator / denominator, 2) if denominator else 0


# ---------------------------------------------------------------- milestones

def progress_headings(text):
    return [line[3:].strip() for line in text.splitlines() if line.startswith("## ")]


def milestones(root):
    data = tomllib.loads((root / MILESTONES).read_text(encoding="utf-8"))
    headings = progress_headings((root / PROGRESS_LOG).read_text(encoding="utf-8"))
    result = []
    if not data.get("milestone"):
        raise ValueError(f"{MILESTONES}: no [[milestone]] entries")
    for milestone in data["milestone"]:
        name = milestone.get("id", "?")
        criteria = milestone.get("criterion", [])
        if not criteria:
            raise ValueError(f"{MILESTONES}: milestone {name} has no criteria")
        rows = []
        for criterion in criteria:
            status = criterion.get("status")
            if status not in STATUSES:
                raise ValueError(f"{MILESTONES}: {name} criterion {criterion.get('text')!r} has status {status!r}, "
                                 f"expected one of {', '.join(STATUSES)}")
            reference = criterion.get("progress", "")
            if status == "verified" and not reference:
                raise ValueError(f"{MILESTONES}: {name} criterion {criterion['text']!r} is verified but names no PROGRESS.md entry")
            if reference and not any(heading.startswith(reference) for heading in headings):
                raise ValueError(f"{MILESTONES}: {name} criterion {criterion['text']!r} cites PROGRESS.md entry "
                                 f"{reference!r}, but no '## ' heading starts with it")
            rows.append({"text": criterion["text"], "status": status, "progress": reference,
                         "evidence": criterion.get("evidence", "")})
        verified = sum(row["status"] == "verified" for row in rows)
        result.append({"id": name, "title": milestone.get("title", ""), "verified": verified, "total": len(rows),
                       "percent": percent(verified, len(rows)), "criteria": rows})
    overall = round(sum(m["verified"] / m["total"] for m in result) / len(result) * 100, 2)
    return {"percent": overall, "milestones": result}


# ---------------------------------------------------------------- titles

def table_rows(lines):
    for line in lines:
        if line.startswith("|"):
            yield [cell.strip() for cell in line.strip().strip("|").split("|")]


def stage_of(status):
    bold = re.search(r"\*\*(.+?)\*\*", status)
    words = (bold.group(1) if bold else status).strip().lower()
    for prefix, score, label in STAGES:
        if words.startswith(prefix):
            return label, score
    raise ValueError(f"{COMPATIBILITY}: status {status[:60]!r} does not start with a known stage "
                     "(Nothing, Loads, Intro, Menu, In game, Playable)")


def titles(root):
    lines = (root / COMPATIBILITY).read_text(encoding="utf-8").splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "## Titles")
    except StopIteration:
        raise ValueError(f"{COMPATIBILITY}: no '## Titles' section") from None
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    rows = list(table_rows(lines[start + 1:end]))
    if len(rows) < 3:
        raise ValueError(f"{COMPATIBILITY}: the Titles table has no rows")
    header = [cell.lower() for cell in rows[0]]
    for column in ("title", "id", "status"):
        if column not in header:
            raise ValueError(f"{COMPATIBILITY}: the Titles table has no {column!r} column")
    title_at, id_at, status_at = header.index("title"), header.index("id"), header.index("status")
    entries = []
    for row in rows[2:]:
        if len(row) <= status_at:
            raise ValueError(f"{COMPATIBILITY}: short row {row[:2]}")
        label, score = stage_of(row[status_at])
        entries.append({"title": row[title_at], "id": row[id_at], "stage": label, "score": score})
    return {"percent": round(sum(e["score"] for e in entries) / len(entries), 2), "titles": entries}


# ---------------------------------------------------------------- GPU tests

def full_runs(logs):
    for summary in logs.glob("*/summary.json"):
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("selectors") == ["all"] and not data.get("meta", {}).get("stopped"):
            yield summary.parent, data


def gpu_snapshot(run, data):
    results, meta = data["results"], data["meta"]
    unallowed = " ".join(data.get("skip_errors", []))
    counts = {kind: sum(r["outcome"] == kind for r in results) for kind in ("passed", "failed", "timeout", "skipped", "not-built")}
    skips = [{"test": r["test"], "reason": r["reason"], "allowlisted": not re.search(rf"\b{re.escape(r['test'])}\b", unallowed)}
             for r in results if r["outcome"] == "skipped"]
    held = sorted(set(data.get("held", [])) | set(meta.get("not_run", [])))
    registered = len(results) + len(held)
    return {
        "run": run.name, "started": meta["started"], "finished": meta["finished"],
        "commit": meta["git"]["head"], "branch": meta["git"]["branch"], "modified_files": meta["git"].get("modified_files", 0),
        "validation": meta.get("environment", {}).get("MTL_SHADER_VALIDATION") == "1",
        "registered": registered, "run_count": len(results), **{k.replace("-", "_"): v for k, v in counts.items()},
        "skipped_allowlisted": sum(s["allowlisted"] for s in skips), "held": held,
        "percent": percent(counts["passed"], registered),
        "skips": skips,
        "failures": [r["test"] for r in results if r["outcome"] in ("failed", "timeout", "not-built")],
    }


def latest_snapshot(logs):
    runs = sorted(full_runs(logs), key=lambda item: item[1]["meta"]["finished"])
    if not runs:
        raise ValueError(f"no full gpu_suite run (selectors ['all']) under {logs}")
    return gpu_snapshot(*runs[-1])


def main_checkout(root):
    """The checkout that owns .mac-work (worktrees under .mac-work/wt share the main one's logs)."""
    common = subprocess.run(["git", "-C", str(root), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            capture_output=True, text=True)
    return Path(common.stdout.strip()).parent if common.returncode == 0 and common.stdout.strip() else root


def gpu_tests(root):
    path = root / GPU_STATUS
    if not path.exists():
        raise ValueError(f"{GPU_STATUS} is missing; run tools/mac/mac_progress.py gpu-snapshot after a full gpu_suite run")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["percent"] = percent(data["passed"], data["registered"])
    return data


# ---------------------------------------------------------------- output

def badge(progress, label, value):
    return progress.badge(label, {"percent": value})


def retarget(page, branch):
    """Point upstream's generated page at this repository's branch instead of main."""
    page = page.replace("/blob/main/", f"/blob/{branch}/")
    return page.replace("<code>main</code>", f"<code>{branch}</code>")


STYLE = """:root{--bg:#fff;--fg:#1f2328;--muted:#59636e;--line:#d1d9e0;--ok:#1a7f37;--part:#9a6700;--open:#cf222e;--bar:#e8ebef}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;--line:#3d444d;--ok:#3fb950;--part:#d29922;--open:#f85149;--bar:#21262d}}
body{background:var(--bg);color:var(--fg);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:1000px;margin:auto;padding:16px;line-height:1.45}
a{color:#4493f8}table{border-collapse:collapse;width:100%;margin:8px 0 16px}th,td{border:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}
th{font-weight:600}.num{text-align:right;white-space:nowrap}.muted{color:var(--muted);font-size:.92em}
.verified{color:var(--ok)}.partial{color:var(--part)}.open{color:var(--open)}
.bar{background:var(--bar);border-radius:3px;height:8px;min-width:80px}.bar span{display:block;height:8px;border-radius:3px;background:var(--ok)}
.badges img{margin-right:6px}.wrap{overflow-x:auto}"""


def bar(value):
    return f'<div class="bar" title="{value}%"><span style="width:{value}%"></span></div>'


def page(upstream, mac, source):
    e = html.escape
    m, g, t = mac["milestones"], mac["gpu"], mac["titles"]
    out = ['<!DOCTYPE html>', '<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
           "<title>AnyPS5-mac progress</title>", f"<style>{STYLE}</style></head><body>",
           "<h1>AnyPS5 on Apple Silicon: progress</h1>",
           '<p class="badges">' + "".join(f'<img src="{name}" alt="{alt}">' for name, alt in (
               ("badge-milestones.svg", "port milestones"), ("badge-metal-tests.svg", "Metal GPU tests"),
               ("badge-titles.svg", "titles"), ("badge-libraries.svg", "libraries"), ("badge-shaders.svg", "shaders"))) + "</p>",
           f'<p class="muted">Generated by <a href="{source}/tools/mac/mac_progress.py">tools/mac/mac_progress.py</a> from '
           f'<code>{e(mac["commit"])}</code> on every push to <code>{BRANCH}</code>. Raw numbers: '
           '<a href="mac-progress.json">mac-progress.json</a>, <a href="progress.json">progress.json</a>.</p>']

    out += ["<h2 id=\"milestones\">Port milestones</h2>",
            f'<p>{m["percent"]}% of the <a href="{source}/docs/mac/MISSION.md">MISSION §8</a> exit criteria verified. '
            "Each milestone counts the same; inside a milestone each criterion counts the same; a partly done criterion counts as 0. "
            f'Source: <a href="{source}/docs/mac/milestones.toml">docs/mac/milestones.toml</a>, where every verified criterion cites '
            f'the <a href="{source}/docs/mac/PROGRESS.md">PROGRESS.md</a> entry that verified it.</p>',
            '<div class="wrap"><table><tr><th>Milestone</th><th class="num">Verified</th><th>Progress</th></tr>']
    for milestone in m["milestones"]:
        out.append(f'<tr><td>{e(milestone["id"])}: {e(milestone["title"])}</td>'
                   f'<td class="num">{milestone["verified"]}/{milestone["total"]}</td><td>{bar(milestone["percent"])}</td></tr>')
    out.append("</table></div>")
    for milestone in m["milestones"]:
        out.append(f'<details><summary>{e(milestone["id"])}: {e(milestone["title"])} ({milestone["verified"]}/{milestone["total"]})</summary>'
                   '<div class="wrap"><table><tr><th>Criterion</th><th>Status</th><th>Evidence</th></tr>')
        for row in milestone["criteria"]:
            reference = f'<br><span class="muted">PROGRESS.md: {e(row["progress"])}</span>' if row["progress"] else ""
            out.append(f'<tr><td>{e(row["text"])}</td><td class="{row["status"]}">{row["status"]}</td>'
                       f'<td>{e(row["evidence"])}{reference}</td></tr>')
        out.append("</table></div></details>")

    out += ["<h2 id=\"metal-gpu-tests\">Metal GPU tests</h2>",
            f'<p>{g["passed"]} of {g["registered"]} registered GPU tests passed ({g["percent"]}%) in the last full '
            f'<code>gpu_suite.py</code> run <code>{e(g["run"])}</code> ({e(g["finished"])}, <code>{e(g["branch"])}</code> at '
            f'<code>{e(g["commit"])}</code>{", Metal API and shader validation on" if g["validation"] else ""}). '
            "Snapshot: <a href=\"" + source + "/docs/mac/gpu-status.json\">docs/mac/gpu-status.json</a>, refreshed on the Mac after a "
            "full run because CI has no Metal device.</p>",
            '<div class="wrap"><table><tr><th>Outcome</th><th class="num">Tests</th></tr>',
            f'<tr><td>passed</td><td class="num">{g["passed"]}</td></tr>',
            f'<tr><td>failed, timed out or not built</td><td class="num">{g["failed"] + g["timeout"] + g["not_built"]}</td></tr>',
            f'<tr><td>skipped, allowlisted (no Metal equivalent or display limit)</td><td class="num">{g["skipped_allowlisted"]}</td></tr>',
            f'<tr><td>skipped, not allowlisted</td><td class="num">{g["skipped"] - g["skipped_allowlisted"]}</td></tr>',
            f'<tr><td>held back (quarantine, not run by the suite)</td><td class="num">{len(g["held"])}</td></tr>',
            "</table></div>"]
    if g["skips"] or g["held"]:
        out.append("<details><summary>Skips and held tests</summary><ul>")
        out += [f'<li><code>{e(s["test"])}</code>{"" if s["allowlisted"] else " (not allowlisted)"}: {e(s["reason"])}</li>' for s in g["skips"]]
        out += [f"<li><code>{e(name)}</code>: held back</li>" for name in g["held"]]
        out.append("</ul></details>")

    out += ["<h2 id=\"titles\">Titles</h2>",
            f'<p>Average stage {t["percent"]}% over {len(t["titles"])} tracked titles in '
            f'<a href="{source}/docs/mac/COMPATIBILITY.md">COMPATIBILITY.md</a>. Scale: Nothing 0, Loads 20, Intro 40, Menu 60, '
            "In game 80, Playable 100 (Playable: 10 minutes with input and audio and correct frames).</p>",
            '<div class="wrap"><table><tr><th>Title</th><th>ID</th><th>Stage</th><th class="num">Score</th></tr>']
    out += [f'<tr><td>{e(x["title"])}</td><td>{e(x["id"])}</td><td>{e(x["stage"])}</td><td class="num">{x["score"]}</td></tr>'
            for x in t["titles"]]
    out.append("</table></div>")

    if upstream:
        out += ["<h2 id=\"upstream\">Upstream metrics</h2>",
                f'<p>System libraries {upstream["libraries"]["done"]}/{upstream["libraries"]["total"]} ({upstream["libraries"]["percent"]}%), '
                f'shader instructions {upstream["shaders"]["done"]}/{upstream["shaders"]["total"]} ({upstream["shaders"]["percent"]}%), '
                f'counted by upstream\'s <a href="{source}/tools/progress.py">tools/progress.py</a> on this tree. They measure the code '
                "shared with upstream AnyPS5 (functions declared and implemented, instructions decoded), not whether they work on the Mac.</p>",
                '<p><a href="progress.html">Library and shader detail</a> · <a href="history.html">History</a></p>',
                '<a href="progress.html"><img src="progress.svg" alt="library and shader map" style="max-width:100%"></a>']
    out.append("</body></html>")
    return "\n".join(out) + "\n"


def collect(root):
    commit = os.environ.get("GITHUB_SHA") or subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                                           capture_output=True, text=True).stdout.strip()
    return {"commit": commit[:9], "milestones": milestones(root), "gpu": gpu_tests(root), "titles": titles(root)}


def write_site(output, root, with_upstream):
    progress = load_upstream_progress()
    output.mkdir(parents=True, exist_ok=True)
    upstream = None
    if with_upstream:
        environment = {**os.environ, "GITHUB_REPOSITORY": os.environ.get("GITHUB_REPOSITORY", "NaitikJoshiPro/AnyPS5-mac")}
        subprocess.run([sys.executable, str(root / "tools" / "progress.py"), str(output)], check=True, env=environment)
        subprocess.run([sys.executable, str(root / "tools" / "progress_history.py"), str(output)], check=True, env=environment, cwd=root)
        for name in ("progress.html", "history.html"):
            path = output / name
            path.write_text(retarget(path.read_text(encoding="utf-8"), BRANCH), encoding="utf-8")
        upstream = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    mac = collect(root)
    repository = os.environ.get("GITHUB_REPOSITORY", "NaitikJoshiPro/AnyPS5-mac")
    source = f"https://github.com/{repository}/blob/{BRANCH}"
    (output / "mac-progress.json").write_text(json.dumps(mac, indent=1) + "\n", encoding="utf-8")
    (output / "badge-milestones.svg").write_text(badge(progress, "port milestones", mac["milestones"]["percent"]), encoding="utf-8")
    (output / "badge-metal-tests.svg").write_text(badge(progress, "Metal GPU tests", mac["gpu"]["percent"]), encoding="utf-8")
    (output / "badge-titles.svg").write_text(badge(progress, "titles", mac["titles"]["percent"]), encoding="utf-8")
    (output / "index.html").write_text(page(upstream, mac, source), encoding="utf-8")
    return mac


def describe(mac):
    m, g, t = mac["milestones"], mac["gpu"], mac["titles"]
    lines = [f'port milestones {m["percent"]}% (' + ", ".join(f'{x["id"]} {x["verified"]}/{x["total"]}' for x in m["milestones"]) + ")",
             f'Metal GPU tests {g["passed"]}/{g["registered"]} ({g["percent"]}%), {g["skipped_allowlisted"]} allowlisted skips, '
             f'{len(g["held"])} held, run {g["run"]} at {g["commit"]}',
             f'titles {t["percent"]}% (' + ", ".join(f'{x["id"]} {x["stage"]}' for x in t["titles"]) + ")"]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=REPO, help="source tree to read (default: this checkout)")
    commands = parser.add_subparsers(dest="command", required=True)
    site = commands.add_parser("site", help="write badges, mac-progress.json and index.html (plus upstream's pages)")
    site.add_argument("output", type=Path)
    site.add_argument("--no-upstream", action="store_true", help="skip tools/progress.py and tools/progress_history.py")
    snapshot = commands.add_parser("gpu-snapshot", help="write docs/mac/gpu-status.json from a full gpu_suite run")
    snapshot.add_argument("--run", type=Path, help="run directory (default: the latest full run under .mac-work/logs/gpu-suite)")
    snapshot.add_argument("--logs", type=Path, help="gpu_suite log directory")
    commands.add_parser("show", help="print the Mac metrics")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.command == "site":
            print(describe(write_site(args.output, root, not args.no_upstream)))
        elif args.command == "show":
            print(describe(collect(root)))
        else:
            if args.run:
                data = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
                if data.get("selectors") != ["all"]:
                    raise ValueError(f"{args.run} is not a full run (selectors {data.get('selectors')})")
                snapshot_data = gpu_snapshot(args.run, data)
            else:
                snapshot_data = latest_snapshot(args.logs or main_checkout(root) / GPU_LOGS)
            (root / GPU_STATUS).write_text(json.dumps(snapshot_data, indent=1) + "\n", encoding="utf-8")
            print(f'{GPU_STATUS}: {snapshot_data["passed"]}/{snapshot_data["registered"]} passed ({snapshot_data["percent"]}%) '
                  f'from {snapshot_data["run"]}')
    except (OSError, ValueError, KeyError) as error:
        print(f"mac_progress: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
