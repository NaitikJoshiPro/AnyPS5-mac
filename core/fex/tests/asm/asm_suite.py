"""Results of FEX's ASM unit tests under aps5-fex-asm: verdicts, baseline and coverage by instruction class.

    python3 asm_suite.py results  <ctest JUnit XML> [variant]
    python3 asm_suite.py baseline <ctest JUnit XML> <baseline file> <description>
    python3 asm_suite.py compare  <ctest JUnit XML> <baseline file>
    python3 asm_suite.py classes  <ctest JUnit XML> <prepared directory> [variant]

Run the suite with JUnit output first, for example
    ctest --test-dir build-arm64 -L '^fex_asm$' -j8 --output-junit <file>
The variant is "default" (ctest tests fex_asm/...) or "full" (fex_asm_full/...).

A verdict is pass, xfail (a listed known failure that fails), skip, fail or xpass (a listed known
failure that passes). The baseline file holds one "<verdict> <variant> <test>" line per test and
variant in the JUnit file; compare checks the variants the JUnit file has and exits 1 when a test that
passed in the baseline does not pass now."""
import collections
import pathlib
import re
import sys
import xml.etree.ElementTree

VERDICT = re.compile(r"^aps5-fex-asm: (PASS|FAIL|XFAIL|XPASS|SKIP) (\S+) \[(\w+)\](?:: (.*))?$")
PREFIXES = {"fex_asm/": "default", "fex_asm_full/": "full"}


def results(junit, variant="default"):
    """Maps each test of the variant to (verdict, detail)."""
    found = {}
    for case in xml.etree.ElementTree.parse(junit).getroot().iter("testcase"):
        name = case.get("name", "")
        prefix = next((prefix for prefix, kind in PREFIXES.items() if name.startswith(prefix) and kind == variant), None)
        if prefix is None:
            continue
        test = name[len(prefix):]
        output = case.findtext("system-out") or ""
        match = next((VERDICT.match(line) for line in output.splitlines() if VERDICT.match(line)), None)
        if match:
            found[test] = (match.group(1).lower(), match.group(4) or "")
        elif case.get("status") == "notrun":
            found[test] = ("skip", "not run by ctest")
        elif case.get("status") == "run":
            found[test] = ("pass", "")
        else:
            found[test] = ("fail", "no verdict from aps5-fex-asm (killed by the ctest timeout?)")
    return found


def all_results(junit):
    """Maps (variant, test) to the verdict, for every variant in the JUnit file."""
    return {(variant, test): verdict for variant in PREFIXES.values() for test, (verdict, _) in results(junit, variant).items()}


def read_baseline(path):
    verdicts = {}
    for line in pathlib.Path(path).read_text().splitlines():
        if line and not line.startswith("#"):
            verdict, variant, test = line.split(" ", 2)
            verdicts[(variant, test)] = verdict
    return verdicts


def write_baseline(junit, path, description):
    verdicts = all_results(junit)
    lines = [f"# {description}"]
    for variant in PREFIXES.values():
        counts = collections.Counter(verdict for (kind, _), verdict in verdicts.items() if kind == variant)
        if counts:
            lines.append(f"# {variant}: " + ", ".join(f"{counts[kind]} {kind}" for kind in ("pass", "xfail", "skip", "fail", "xpass")))
    lines.append("# One line per test and variant: <verdict> <variant> <test>. Verdicts: pass, xfail (a listed known failure),"
                 " skip, fail, xpass.")
    lines += [f"{verdict} {variant} {test}" for (variant, test), verdict in sorted(verdicts.items())]
    pathlib.Path(path).write_text("\n".join(lines) + "\n")
    print("\n".join(line[2:] for line in lines[1:-len(verdicts) - 1]))
    return 0


def compare(junit, path):
    now = all_results(junit)
    present = {variant for variant, _ in now}
    before = {key: verdict for key, verdict in read_baseline(path).items() if key[0] in present}
    for variant in sorted({variant for variant, _ in read_baseline(path)} - present):
        print(f"note: the JUnit file has no {variant} results; that part of the baseline is not compared")
    lost = sorted(test for test, verdict in before.items() if verdict == "pass" and now.get(test) != "pass")
    gained = sorted(test for test, verdict in now.items() if verdict == "pass" and before.get(test) != "pass")
    changed = sorted(test for test in set(before) | set(now) if before.get(test) != now.get(test) and test not in lost and test not in gained)
    for key in lost:
        print(f"regression: {key[1]} [{key[0]}]: {before[key]} -> {now.get(key, 'missing')}")
    for key in gained:
        print(f"now passes: {key[1]} [{key[0]}]: {before.get(key, 'missing')} -> pass")
    for key in changed:
        print(f"changed: {key[1]} [{key[0]}]: {before.get(key, 'missing')} -> {now.get(key, 'missing')}")
    print(f"{len(before)} tests in the baseline, {len(now)} now: {len(lost)} regressions, {len(gained)} newly passing, {len(changed)} other changes")
    return 1 if lost else 0


FMA = re.compile(r"^vf(n?m(add|sub)(sub|add)?|madd|msub)(132|213|231)")
AVX2_ONLY = {"vpbroadcastb", "vpbroadcastw", "vpbroadcastd", "vpbroadcastq", "vbroadcasti128", "vperm2i128", "vpermd", "vpermq",
             "vpermps", "vpermpd", "vinserti128", "vextracti128", "vpblendd", "vpsllvd", "vpsllvq", "vpsrlvd", "vpsrlvq", "vpsravd",
             "vpgatherdd", "vpgatherdq", "vpgatherqd", "vpgatherqq", "vgatherdps", "vgatherdpd", "vgatherqps", "vgatherqpd",
             "vpmaskmovd", "vpmaskmovq"}
SSE = set("addps addss andnps andps cmpps cmpss comiss cvtpi2ps cvtps2pi cvtsi2ss cvtss2si cvttps2pi cvttss2si divps divss ldmxcsr "
          "maxps maxss minps minss movaps movhlps movhps movlhps movlps movmskps movntps movss movups mulps mulss orps rcpps rcpss "
          "rsqrtps rsqrtss shufps sqrtps sqrtss stmxcsr subps subss ucomiss unpckhps unpcklps xorps pavgb pavgw pextrw pinsrw "
          "pmaxsw pmaxub pminsw pminub pmovmskb pmulhuw psadbw pshufw maskmovq movntq sfence cmpeqps cmpltps cmpleps "
          "cmpunordps cmpneqps cmpnltps cmpnleps cmpordps cmpeqss cmpltss cmpless cmpunordss cmpneqss cmpnltss cmpnless cmpordss".split())
SSE2 = set("addpd addsd andnpd andpd cmppd comisd cvtdq2pd cvtdq2ps cvtpd2dq cvtpd2pi cvtpd2ps cvtpi2pd cvtps2dq cvtps2pd cvtsd2si "
           "cvtsd2ss cvtsi2sd cvtss2sd cvttpd2dq cvttpd2pi cvttps2dq cvttsd2si divpd divsd maxpd maxsd minpd minsd movapd movhpd "
           "movlpd movmskpd movupd mulpd mulsd orpd shufpd sqrtpd sqrtsd subpd subsd ucomisd unpckhpd unpcklpd xorpd movdq2q "
           "movdqa movdqu movq2dq pmuludq pshufd pshufhw pshuflw pslldq psrldq punpckhqdq punpcklqdq maskmovdqu movntdq movnti "
           "movntpd lfence mfence clflush pause cmpeqpd cmpltpd cmplepd cmpunordpd cmpneqpd cmpnltpd cmpnlepd cmpordpd cmpeqsd "
           "cmpltsd cmplesd cmpunordsd cmpneqsd cmpnltsd cmpnlesd cmpordsd".split())
SSE3 = set("addsubpd addsubps haddpd haddps hsubpd hsubps lddqu movddup movshdup movsldup fisttp monitor mwait".split())
SSSE3 = set("pabsb pabsd pabsw palignr phaddd phaddsw phaddw phsubd phsubsw phsubw pmaddubsw pmulhrsw pshufb psignb psignd psignw".split())
SSE41 = set("blendpd blendps blendvpd blendvps dppd dpps extractps insertps movntdqa mpsadbw packusdw pblendvb pblendw pcmpeqq "
            "pextrb pextrd pextrq phminposuw pinsrb pinsrd pinsrq pmaxsb pmaxsd pmaxud pmaxuw pminsb pminsd pminud pminuw pmovsxbd "
            "pmovsxbq pmovsxbw pmovsxdq pmovsxwd pmovsxwq pmovzxbd pmovzxbq pmovzxbw pmovzxdq pmovzxwd pmovzxwq pmuldq pmulld ptest "
            "roundpd roundps roundsd roundss".split())
SSE42 = set("crc32 pcmpestri pcmpestrm pcmpgtq pcmpistri pcmpistrm".split())
SSE4A = set("extrq insertq movntsd movntss".split())
CRYPTO = set("aesdec aesdeclast aesenc aesenclast aesimc aeskeygenassist pclmulqdq pclmullqlqdq pclmulhqlqdq pclmullqhqdq "
             "pclmulhqhqdq sha1msg1 sha1msg2 sha1nexte sha1rnds4 sha256msg1 sha256msg2 sha256rnds2".split())
BMI1 = set("andn bextr blsi blsmsk blsr tzcnt".split())
BMI2 = set("bzhi mulx pdep pext rorx sarx shlx shrx".split())
ABM = set("lzcnt popcnt".split())
THREEDNOW = set("femms pavgusb pf2id pf2iw pfacc pfadd pfcmpeq pfcmpge pfcmpgt pfmax pfmin pfmul pfnacc pfpnacc pfrcp pfrcpit1 "
                "pfrcpit2 pfrsqit1 pfrsqrt pfsub pfsubr pi2fd pi2fw pmulhrw pswapd".split())
SYSTEM = set("cpuid rdtsc rdtscp rdpid rdpmc xgetbv xsave xsave64 xsaveopt xsavec xrstor xrstor64 fxsave fxsave64 fxrstor fxrstor64 "
             "clflushopt clwb clzero rdrand rdseed rdfsbase rdgsbase wrfsbase wrgsbase ud2 int3 int int1 into syscall sysenter in out "
             "cli sti hlt prefetch prefetchw prefetcht0 prefetcht1 prefetcht2 prefetchnta lar lsl sgdt sidt sldt str smsw verr "
             "verw arpl ud0 ud1 wbinvd invd rsm cpuid".split())
STRING = re.compile(r"^(movs|cmps|stos|scas|lods|ins|outs)[bwdq]?$")
FLAG_READERS = re.compile(r"^(pushf[wdq]?|lahf|set[a-z]+|cmov[a-z]+|j(?!mp$)[a-z]+|adc|sbb|rcl|rcr|adcx|adox|loopn?[ez]|fcmov[a-z]+|salc)$")
SETUP = {"mov", "lea", "hlt", "ret", "jmp", "nop", "push", "pop", "xor", "call"}
CLASSES = ["integer", "flags", "string", "atomic", "x87", "MMX", "3DNow!", "SSE", "SSE2", "SSE3", "SSSE3", "SSE4.1", "SSE4.2", "SSE4a",
           "AES/PCLMUL/SHA", "AVX", "AVX2", "FMA", "F16C", "BMI1", "BMI2", "LZCNT/POPCNT", "MOVBE", "system"]


def instruction_class(token):
    prefix, _, rest = token.rpartition(":")
    mnemonic, _, width = rest.partition("@")
    found = set()
    if prefix == "lock" or mnemonic in ("cmpxchg8b", "cmpxchg16b"):
        found.add("atomic")
    if STRING.match(mnemonic) and not width:
        found.add("string")
        return found
    if FLAG_READERS.match(mnemonic):
        found.add("flags")
    if mnemonic in THREEDNOW:
        return found | {"3DNow!"}
    if mnemonic.startswith("f") and mnemonic not in ("fxsave", "fxsave64", "fxrstor", "fxrstor64") or mnemonic in ("wait", "fwait"):
        return found | ({"SSE3"} if mnemonic == "fisttp" else {"x87"})
    if mnemonic == "movbe":
        return found | {"MOVBE"}
    if mnemonic in BMI1:
        return found | {"BMI1"}
    if mnemonic in BMI2:
        return found | {"BMI2"}
    if mnemonic in ABM:
        return found | {"LZCNT/POPCNT"}
    if mnemonic in SYSTEM:
        return found | {"system"}
    if mnemonic.startswith("v") and mnemonic not in ("verr", "verw"):
        if FMA.match(mnemonic):
            return found | {"FMA"}
        if mnemonic in ("vcvtph2ps", "vcvtps2ph"):
            return found | {"F16C"}
        if mnemonic.startswith(("vaes", "vpclmul", "vsha")):
            return found | {"AES/PCLMUL/SHA", "AVX"}
        if mnemonic in AVX2_ONLY or (mnemonic.startswith("vp") and width == "ymm"):
            return found | {"AVX2"}
        return found | {"AVX"}
    if mnemonic in CRYPTO:
        return found | {"AES/PCLMUL/SHA"}
    for table, name in ((SSE42, "SSE4.2"), (SSE41, "SSE4.1"), (SSSE3, "SSSE3"), (SSE3, "SSE3"), (SSE4A, "SSE4a"), (SSE2, "SSE2"), (SSE, "SSE")):
        if mnemonic in table:
            if name in ("SSSE3",) and width == "mm":
                return found | {"SSSE3", "MMX"}
            return found | {name}
    if mnemonic in ("movsd", "cmpsd") and width:
        return found | {"SSE2"}
    if width == "mm" or mnemonic == "emms":
        return found | {"MMX"}
    if width == "xmm" and mnemonic.startswith(("p", "movd", "movq", "packs", "packu", "punpck")):
        return found | {"SSE2"}
    if width:
        return found | {"SSE2"}
    return found | {"integer"}


def test_classes(tokens):
    classes = set()
    meaningful = [token for token in tokens if token.rpartition(":")[2].partition("@")[0] not in SETUP]
    for token in meaningful or tokens:
        classes |= instruction_class(token)
    return classes


def report_classes(junit, prepared, variant="default"):
    verdicts = results(junit, variant)
    per_class = {name: collections.Counter() for name in CLASSES}
    unclassified = collections.Counter()
    for test, (verdict, _) in verdicts.items():
        config = pathlib.Path(prepared) / (test + ".cfg")
        tokens = []
        if config.exists():
            for line in config.read_text().splitlines():
                if line.startswith("instructions "):
                    tokens = line.split()[1:]
        if not tokens:
            unclassified[verdict] += 1
            continue
        for name in test_classes(tokens):
            per_class[name][verdict] += 1
    print(f"{'class':16} {'tests':>6} {'pass':>6} {'xfail':>6} {'fail':>6} {'skip':>6} {'pass rate':>10}")
    for name in CLASSES:
        counts = per_class[name]
        total = sum(counts.values())
        ran = total - counts["skip"]
        rate = f"{100.0 * counts['pass'] / ran:.1f}%" if ran else "-"
        print(f"{name:16} {total:6} {counts['pass']:6} {counts['xfail']:6} {counts['fail'] + counts['xpass']:6} {counts['skip']:6} {rate:>10}")
    total = collections.Counter(verdict for verdict, _ in verdicts.values())
    ran = sum(total.values()) - total["skip"]
    print(f"{'all tests':16} {sum(total.values()):6} {total['pass']:6} {total['xfail']:6} {total['fail'] + total['xpass']:6} "
          f"{total['skip']:6} {100.0 * total['pass'] / ran if ran else 0:9.1f}%")
    print(f"not classified (not assembled: disabled or unmovable): {sum(unclassified.values())}")
    return 0


def main(arguments):
    if len(arguments) >= 2 and arguments[0] == "results":
        for test, (verdict, detail) in sorted(results(arguments[1], *arguments[2:3]).items()):
            print(f"{verdict}\t{test}\t{detail}")
        return 0
    if len(arguments) == 4 and arguments[0] == "baseline":
        return write_baseline(arguments[1], arguments[2], arguments[3])
    if len(arguments) == 3 and arguments[0] == "compare":
        return compare(arguments[1], arguments[2])
    if len(arguments) in (3, 4) and arguments[0] == "classes":
        return report_classes(arguments[1], arguments[2], *arguments[3:4])
    sys.stderr.write(__doc__.split("\n\n")[1] + "\n")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
