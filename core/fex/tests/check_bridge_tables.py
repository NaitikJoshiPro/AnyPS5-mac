"""Check the bridge's tables against the libraries' sources.

    python3 check_bridge_tables.py <Bridge.cpp> <libraries source directory>

On arm64 the bridge cannot tell from a library function how to pass its arguments: a variadic
function takes its variadic arguments on the stack, and a long double is the guest's x87 value.
Bridge.cpp lists both kinds by name, so a library export that is variadic or uses a long double
must be listed there, with the right number of fixed arguments."""
import pathlib
import re
import sys

DEFINITION = re.compile(r"^([^\n;{}]*?)\b(\w+_nid_postfix)\s*\(([^;{}()]*(?:\([^()]*\)[^;{}()]*)*)\)\s*(?:noexcept\s*)?\{", re.M)
LONG_DOUBLE = re.compile(r"\blong double\b|\bGuestLongDouble\b")


def parameters(text):
    """The parameter list split at its top-level commas."""
    depth, current, result = 0, "", []
    for character in text:
        if character in "(<[":
            depth += 1
        elif character in ")>]":
            depth -= 1
        if character == "," and depth == 0:
            result.append(current.strip())
            current = ""
        else:
            current += character
    if current.strip() and current.strip() != "void":
        result.append(current.strip())
    return result


def main():
    bridge = pathlib.Path(sys.argv[1]).read_text()
    variadic = {name: int(fixed) for name, fixed in re.findall(r'\{"(\w+_nid_postfix)",\s*\w+,\s*(\d+)\}', bridge)}
    x87 = set(re.findall(r'"(\w+_nid_postfix)"', bridge.split("X87Results[]", 1)[1].split(";", 1)[0]))

    problems = []
    defined_variadic, defined_x87 = set(), set()
    for path in sorted(pathlib.Path(sys.argv[2]).rglob("*")):
        if path.suffix not in (".c", ".cpp", ".mm") or "tests" in path.parts:
            continue
        for match in DEFINITION.finditer(path.read_text(errors="ignore")):
            prefix, name, listed = match.groups()
            where = f"{path}:{match.string.count(chr(10), 0, match.start(2)) + 1}"
            arguments = parameters(listed)
            if arguments and arguments[-1] == "...":
                defined_variadic.add(name)
                if name not in variadic:
                    problems.append(f"{where}: {name} is variadic but not in VariadicExports")
                elif variadic[name] != len(arguments) - 1:
                    problems.append(f"{where}: {name} has {len(arguments) - 1} fixed arguments, VariadicExports says {variadic[name]}")
            if LONG_DOUBLE.search(prefix):
                defined_x87.add(name)
                if name not in x87:
                    problems.append(f"{where}: {name} returns a long double but is not in X87Results")
            if any(LONG_DOUBLE.search(argument) for argument in arguments):
                problems.append(f"{where}: {name} takes a long double, which the bridge cannot pass")
    problems += [f"VariadicExports lists {name}, which no library defines as variadic" for name in sorted(set(variadic) - defined_variadic)]
    problems += [f"X87Results lists {name}, which no library defines as returning a long double" for name in sorted(x87 - defined_x87)]

    for problem in problems:
        print(problem)
    if problems:
        return 1
    print(f"{len(defined_variadic)} variadic and {len(defined_x87)} long double exports are in the bridge's tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
