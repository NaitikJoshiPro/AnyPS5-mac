#!/usr/bin/env python3
"""Derive the bridge's signature table from the arm64 prx library sources.

    gen_bridge_signatures.py generate --compile-commands FILE --root DIR --cache DIR --json FILE --inc FILE [--jobs N]
    gen_bridge_signatures.py check --json FILE --bridge Bridge.cpp [--verbose]
    gen_bridge_signatures.py verify --json FILE --work DIR [--compile-commands FILE --root DIR]

generate: libclang (APS5_LIBCLANG, or the libclang.dylib beside the compiler that compile_commands.json
names) parses every translation unit of a prx library target (it defines <target>_EXPORTS, lives under
core/libs/prx, or under core/fex/tests/prx for the bridge's test library, and not in a tests directory). Each extern "C" function that a library defines with
external linkage and default visibility is an export. Its result and parameters are classified
(integer, pointer, function pointer with the callback's own signature, float, double, long double,
GuestLongDouble, __int128, vector, empty, struct with its layout and whether C++ passes it through a
hidden reference), placed under the guest's x86-64 SysV convention and under Apple's arm64 convention,
and compared with the bridge's positional mapping: rdi..r9 to x0..x5, the first two guest stack slots
to x6 and x7, xmm to v, and the next eight 8-byte stack slots copied to the native stack. Exports named
*_nid_no_patch are host-only and left out. --json gets everything, --inc the table that
core/fex/src/BridgeSignatures.cpp compiles. Parse results are cached per translation unit in --cache
and reused while its arguments, every repository file it includes, libclang and the parsing part of this
script are unchanged.

check: fails when an export cannot be classified, when two definitions of one export differ, or when
Bridge.cpp's VariadicExports or X87Results disagree with the generated data, and prints every export
that needs more than positional marshalling.

verify: compiles a function of every distinct export and callback shape for x86_64-sie-ps5 and for
arm64-apple-macos with Apple clang, and fails when the argument registers, stack ranges or result
registers in the MIR after instruction selection differ from the generator's. With --compile-commands
it also compiles the library units whose exports pass structs, GuestLongDouble or __int128 by value and
checks those exports on arm64, which also checks how their types were classified.
"""
import argparse
import ctypes
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

SKIP_FUNCTION_BODIES = 0x40
DISPLAY_SOURCE_LOCATION = 0x01

CURSOR_UNEXPOSED_DECL = 1
CURSOR_UNION = 3
CURSOR_FIELD = 6
CURSOR_FUNCTION = 8
CURSOR_METHOD = 21
CURSOR_NAMESPACE = 22
CURSOR_LINKAGE_SPEC = 23
CURSOR_CONSTRUCTOR = 24
CURSOR_DESTRUCTOR = 25
CURSOR_BASE = 44

LINKAGE_EXTERNAL = 4
VISIBILITY_DEFAULT = 3
SEVERITY_ERROR = 3

TYPE_UNEXPOSED = 1
TYPE_VOID = 2
TYPE_BOOL = 3
UNSIGNED_TYPES = {4, 5, 6, 7, 8, 9, 10, 11}
TYPE_UINT128 = 12
SIGNED_TYPES = {13, 14, 15, 16, 17, 18, 19}
TYPE_INT128 = 20
TYPE_FLOAT = 21
TYPE_DOUBLE = 22
TYPE_LONG_DOUBLE = 23
TYPE_NULLPTR = 24
TYPE_COMPLEX = 100
POINTER_TYPES = {101, 102, 103, 104, 109}
TYPE_RECORD = 105
TYPE_ENUM = 106
TYPE_FUNCTION_NO_PROTO = 110
TYPE_FUNCTION_PROTO = 111
TYPE_CONSTANT_ARRAY = 112
TYPE_VECTOR = 113
TYPE_INCOMPLETE_ARRAY = 114
ARRAY_TYPES = {TYPE_CONSTANT_ARRAY, TYPE_INCOMPLETE_ARRAY, 115, 116}
TYPE_MEMBER_POINTER = 117
TYPE_EXT_VECTOR = 176

GUEST_LONG_DOUBLE = "X87Extended"
HOST_ONLY = ("_nid_no_patch", "_nid_no_patch_cut")
FLATTENED_RECORD_BYTES = 64
CALLBACK_DEPTH = 2

GUEST_REGISTERS = ("rdi", "rsi", "rdx", "rcx", "r8", "r9")
GUEST_RESULT_REGISTERS = {"rax": "x0", "rdx": "x1", "xmm0": "v0", "xmm1": "v1"}
POSITIONAL_STACK_BYTES = 80

REASONS = {
    "variadic": ("Variadic", "variadic: the fixed arguments are positional, the variadic ones need VariadicExports"),
    "x87-result": ("X87Result", "returns GuestLongDouble: x0:x1 on arm64, st(0) on the guest (X87Results)"),
    "long-double": ("LongDouble", "long double: the guest's x87 80-bit value in memory or st(0), a double in a d register on arm64"),
    "x87-argument": ("X87Argument", "GuestLongDouble argument: in memory on the guest, a register pair on arm64"),
    "stack-layout": ("StackLayout", "a stack argument: arm64 packs stack arguments at their natural size, or the guest's slot is not copied"),
    "register-shift": ("RegisterShift", "an argument in another register: an earlier argument uses registers differently"),
    "struct-registers": ("StructRegisters", "a struct in registers: SysV eightbytes against arm64 HFA members or general registers"),
    "struct-memory": ("StructMemory", "a struct over 16 bytes by value: copied to the guest stack, passed by pointer on arm64"),
    "sret": ("HiddenResult", "a result through a hidden pointer: rdi on the guest (returned in rax), x8 on arm64"),
    "struct-result": ("StructResult", "a struct result in registers that the positional mapping places wrongly"),
    "int128": ("Int128", "__int128: a register pair that positional mapping places differently (or splits from the stack)"),
    "unclassified": ("Unclassified", "a type the generator cannot classify"),
}

KIND_NAMES = {"void": "Void", "int128": "Int128", "pointer": "Pointer", "function": "Function", "float": "Float",
              "double": "Double", "long double": "LongDouble", "x87": "X87", "vector": "Vector", "record": "Record",
              "empty": "Empty", "unsupported": "Unsupported"}
LEAF_NAMES = {"int": "Integer", "float": "Float", "double": "Double", "long double": "LongDouble", "x87": "X87",
              "vector": "Vector"}
LEAF_KINDS = {"int": "int", "pointer": "int", "function": "int", "int128": "int", "float": "float", "double": "double",
              "long double": "long double", "vector": "vector", "x87": "x87"}


class Unclassified(Exception):
    """A type the generator cannot describe."""


class Cursor(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int), ("xdata", ctypes.c_int), ("data", ctypes.c_void_p * 3)]


class Type(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int), ("data", ctypes.c_void_p * 2)]


class String(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("flags", ctypes.c_uint)]


class Location(ctypes.Structure):
    _fields_ = [("pointers", ctypes.c_void_p * 2), ("value", ctypes.c_uint)]


class Range(ctypes.Structure):
    _fields_ = [("pointers", ctypes.c_void_p * 2), ("begin", ctypes.c_uint), ("end", ctypes.c_uint)]


CHILD_VISITOR = ctypes.CFUNCTYPE(ctypes.c_int, Cursor, Cursor, ctypes.c_void_p)
FIELD_VISITOR = ctypes.CFUNCTYPE(ctypes.c_int, Cursor, ctypes.c_void_p)
INCLUSION_VISITOR = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p)

_P, _U, _I, _L = ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_longlong
_POSITION = [Location, ctypes.POINTER(_P), ctypes.POINTER(_U), ctypes.POINTER(_U), ctypes.POINTER(_U)]
FUNCTIONS = {
    "createIndex": (_P, [_I, _I]),
    "disposeIndex": (None, [_P]),
    "parseTranslationUnit2": (_I, [_P, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), _I, _P, _U, _U, ctypes.POINTER(_P)]),
    "disposeTranslationUnit": (None, [_P]),
    "getTranslationUnitCursor": (Cursor, [_P]),
    "visitChildren": (_U, [Cursor, CHILD_VISITOR, _P]),
    "getCString": (ctypes.c_char_p, [String]),
    "disposeString": (None, [String]),
    "getCursorSpelling": (String, [Cursor]),
    "getCursorUSR": (String, [Cursor]),
    "Cursor_getMangling": (String, [Cursor]),
    "getCursorLinkage": (_I, [Cursor]),
    "isCursorDefinition": (_U, [Cursor]),
    "getCursorVisibility": (_I, [Cursor]),
    "getCursorType": (Type, [Cursor]),
    "getCursorResultType": (Type, [Cursor]),
    "Cursor_getNumArguments": (_I, [Cursor]),
    "Cursor_getArgument": (Cursor, [Cursor, _U]),
    "isFunctionTypeVariadic": (_U, [Type]),
    "getCanonicalType": (Type, [Type]),
    "getTypeSpelling": (String, [Type]),
    "Type_getSizeOf": (_L, [Type]),
    "Type_getAlignOf": (_L, [Type]),
    "getPointeeType": (Type, [Type]),
    "getResultType": (Type, [Type]),
    "getNumArgTypes": (_I, [Type]),
    "getArgType": (Type, [Type, _U]),
    "getTypeDeclaration": (Cursor, [Type]),
    "getEnumDeclIntegerType": (Type, [Cursor]),
    "getElementType": (Type, [Type]),
    "getNumElements": (_L, [Type]),
    "getArraySize": (_L, [Type]),
    "Type_visitFields": (_U, [Type, FIELD_VISITOR, _P]),
    "Cursor_getOffsetOfField": (_L, [Cursor]),
    "Cursor_isBitField": (_U, [Cursor]),
    "getFieldDeclBitWidth": (_I, [Cursor]),
    "getOffsetOfBase": (_L, [Cursor, Cursor]),
    "isVirtualBase": (_U, [Cursor]),
    "getSpecializedCursorTemplate": (Cursor, [Cursor]),
    "Cursor_isNull": (_I, [Cursor]),
    "CXXMethod_isVirtual": (_U, [Cursor]),
    "CXXMethod_isDefaulted": (_U, [Cursor]),
    "CXXMethod_isDeleted": (_U, [Cursor]),
    "CXXConstructor_isCopyConstructor": (_U, [Cursor]),
    "CXXConstructor_isMoveConstructor": (_U, [Cursor]),
    "getCursorLocation": (Location, [Cursor]),
    "getCursorExtent": (Range, [Cursor]),
    "getRangeEnd": (Location, [Range]),
    "getSpellingLocation": (None, _POSITION),
    "getExpansionLocation": (None, _POSITION),
    "Location_isInSystemHeader": (_I, [Location]),
    "getFileName": (String, [_P]),
    "getInclusions": (None, [_P, INCLUSION_VISITOR, _P]),
    "getNumDiagnostics": (_U, [_P]),
    "getDiagnostic": (_P, [_P, _U]),
    "getDiagnosticSeverity": (_I, [_P]),
    "formatDiagnostic": (String, [_P, _U]),
    "disposeDiagnostic": (None, [_P]),
}


class LibClang:
    """The libclang C API through ctypes."""

    def __init__(self, path):
        library = ctypes.CDLL(path)
        for name, (result, arguments) in FUNCTIONS.items():
            function = getattr(library, "clang_" + name)
            function.restype = result
            function.argtypes = arguments
            setattr(self, name, function)

    def text(self, string):
        value = self.getCString(string)
        self.disposeString(string)
        return value.decode("utf-8", "replace") if value else ""

    def position(self, location, expansion=False):
        file, line, column, offset = _P(), _U(), _U(), _U()
        reader = self.getExpansionLocation if expansion else self.getSpellingLocation
        reader(location, ctypes.byref(file), ctypes.byref(line), ctypes.byref(column), ctypes.byref(offset))
        return (self.text(self.getFileName(file)) if file.value else ""), line.value, offset.value

    def children(self, cursor, skip_system=False):
        found = []

        @CHILD_VISITOR
        def visit(child, parent, data):
            if not skip_system or not self.Location_isInSystemHeader(self.getCursorLocation(child)):
                found.append(Cursor.from_buffer_copy(child))
            return 1

        self.visitChildren(cursor, visit, None)
        return found

    def fields(self, record):
        found = []

        @FIELD_VISITOR
        def visit(field, data):
            found.append(Cursor.from_buffer_copy(field))
            return 1

        self.Type_visitFields(record, visit, None)
        return found


def opens_body(data, offset, in_macro):
    """True when the declaration ending at offset is followed by a function body, False when it is not,
    None when a macro definition ends first."""
    depth, position = 0, offset
    while position < len(data):
        if data.startswith(b"//", position):
            position = data.find(b"\n", position)
            if position < 0:
                return False
            continue
        if data.startswith(b"/*", position):
            end = data.find(b"*/", position + 2)
            if end < 0:
                return False
            position = end + 2
            continue
        character = data[position:position + 1]
        if character == b"\n" and in_macro and not data[:position].rstrip(b" \t\r").endswith(b"\\"):
            return None
        if data.startswith(b"->", position):
            position += 2
            continue
        if character in b"([<":
            depth += 1
        elif character in b")]>":
            depth -= 1
        elif depth <= 0 and character == b"{":
            return True
        elif depth <= 0 and character in (b";", b"=", b","):
            return False
        position += 1
    return False


class UnitReader:
    """Reads the exports of one parsed translation unit."""

    def __init__(self, clang, root, full):
        self.clang = clang
        self.full = full
        self.libraries = (str(Path(root) / "core" / "libs") + os.sep, str(Path(root) / "core" / "fex" / "tests" / "prx") + os.sep)
        self.root = str(Path(root)) + os.sep
        self.texts = {}
        self.indirect_records = {}

    def exports(self, unit):
        found = []
        self.visit(self.clang.getTranslationUnitCursor(unit), found)
        return found

    def visit(self, parent, found):
        for cursor in self.clang.children(parent, skip_system=True):
            if cursor.kind in (CURSOR_NAMESPACE, CURSOR_LINKAGE_SPEC, CURSOR_UNEXPOSED_DECL):
                self.visit(cursor, found)
            elif cursor.kind == CURSOR_FUNCTION:
                export = self.function(cursor)
                if export is not None:
                    found.append(export)

    def function(self, cursor):
        clang = self.clang
        file, line, _ = clang.position(clang.getCursorLocation(cursor))
        if not file.startswith(self.libraries):
            return None
        if clang.getCursorLinkage(cursor) != LINKAGE_EXTERNAL or clang.getCursorVisibility(cursor) != VISIBILITY_DEFAULT:
            return None
        name = clang.text(clang.getCursorSpelling(cursor))
        if clang.text(clang.Cursor_getMangling(cursor)) != "_" + name or not self.defines(cursor):
            return None
        export = {"name": name, "location": f"{os.path.relpath(file, self.root)}:{line}"}
        try:
            export.update(self.signature(cursor))
        except Unclassified as error:
            export.update({"variadic": False, "result": {"kind": "unsupported", "reason": str(error)}, "parameters": []})
        return export

    def text_of(self, file):
        if file not in self.texts:
            self.texts[file] = Path(file).read_bytes()
        return self.texts[file]

    def defines(self, cursor):
        clang = self.clang
        if self.full:
            return bool(clang.isCursorDefinition(cursor))
        end = clang.getRangeEnd(clang.getCursorExtent(cursor))
        spelled = clang.position(end)
        expanded = clang.position(end, expansion=True)
        if spelled != expanded and spelled[0]:
            verdict = opens_body(self.text_of(spelled[0]), spelled[2], True)
            if verdict is not None:
                return verdict
        return bool(expanded[0]) and opens_body(self.text_of(expanded[0]), expanded[2], False) is True

    def signature(self, cursor):
        clang = self.clang
        parameters = [self.value(clang.getCursorType(clang.Cursor_getArgument(cursor, index)), 0)
                      for index in range(clang.Cursor_getNumArguments(cursor))]
        return {"variadic": bool(clang.isFunctionTypeVariadic(clang.getCursorType(cursor))),
                "result": self.value(clang.getCursorResultType(cursor), 0), "parameters": parameters}

    def callback(self, function_type, depth):
        clang = self.clang
        if function_type.kind != TYPE_FUNCTION_PROTO:
            raise Unclassified("a function pointer without a prototype")
        return {"variadic": bool(clang.isFunctionTypeVariadic(function_type)),
                "result": self.value(clang.getResultType(function_type), depth),
                "parameters": [self.value(clang.getArgType(function_type, index), depth)
                               for index in range(clang.getNumArgTypes(function_type))]}

    def value(self, given, depth):
        clang = self.clang
        canonical = clang.getCanonicalType(given)
        kind = canonical.kind
        spelling = clang.text(clang.getTypeSpelling(canonical))
        size, align = clang.Type_getSizeOf(canonical), clang.Type_getAlignOf(canonical)
        if kind == TYPE_VOID:
            return {"kind": "void", "type": spelling}
        if kind == TYPE_BOOL or kind in UNSIGNED_TYPES or (kind == TYPE_UNEXPOSED and spelling == "char8_t"):
            return {"kind": "int", "type": spelling, "size": size, "align": align, "signed": False}
        if kind in SIGNED_TYPES:
            return {"kind": "int", "type": spelling, "size": size, "align": align, "signed": True}
        if kind in (TYPE_INT128, TYPE_UINT128):
            return {"kind": "int128", "type": spelling, "size": size, "align": align, "signed": kind == TYPE_INT128}
        if kind in (TYPE_FLOAT, TYPE_DOUBLE, TYPE_LONG_DOUBLE):
            name = {TYPE_FLOAT: "float", TYPE_DOUBLE: "double", TYPE_LONG_DOUBLE: "long double"}[kind]
            return {"kind": name, "type": spelling, "size": size, "align": align}
        if kind == TYPE_NULLPTR or kind in ARRAY_TYPES:
            return {"kind": "pointer", "type": spelling, "size": 8, "align": 8}
        if kind in POINTER_TYPES or kind in (TYPE_FUNCTION_PROTO, TYPE_FUNCTION_NO_PROTO):
            pointee = canonical if kind not in POINTER_TYPES else clang.getCanonicalType(clang.getPointeeType(canonical))
            if pointee.kind in (TYPE_FUNCTION_PROTO, TYPE_FUNCTION_NO_PROTO):
                value = {"kind": "function", "type": spelling, "size": 8, "align": 8}
                if depth < CALLBACK_DEPTH:
                    value["callback"] = self.callback(pointee, depth + 1)
                return value
            return {"kind": "pointer", "type": spelling, "size": 8, "align": 8}
        if kind == TYPE_ENUM:
            value = self.value(clang.getEnumDeclIntegerType(clang.getTypeDeclaration(canonical)), depth)
            value["type"] = spelling
            return value
        if kind == TYPE_MEMBER_POINTER and size == 8:
            return {"kind": "int", "type": spelling, "size": 8, "align": 8, "signed": True}
        if kind == TYPE_MEMBER_POINTER and size == 16:
            return {"kind": "record", "type": spelling, "size": 16, "align": 8, "indirect": False, "union": False,
                    "leaves": [[0, "int", 8, 8], [8, "int", 8, 8]]}
        if kind == TYPE_RECORD:
            return self.record(canonical, spelling, size, align)
        if kind in (TYPE_VECTOR, TYPE_EXT_VECTOR):
            element = self.value(clang.getElementType(canonical), CALLBACK_DEPTH)
            count = clang.getNumElements(canonical)
            if size not in (8, 16) or count < 2:
                raise Unclassified(f"the vector {spelling}: only 8 and 16 bytes with several elements are placed")
            return {"kind": "vector", "type": spelling, "size": size, "align": align,
                    "element": LEAF_KINDS.get(element["kind"], "int"), "count": count}
        if kind == TYPE_COMPLEX:
            element = self.value(clang.getElementType(canonical), CALLBACK_DEPTH)
            if element["kind"] == "long double":
                # The guest's is two x87 values: 32 bytes in memory, a result in st(0) and st(1) (COMPLEX_X87).
                raise Unclassified(f"{spelling}: its long double parts differ in size between the guest and arm64")
            leaf, part = LEAF_KINDS[element["kind"]], element["size"]
            return {"kind": "record", "type": spelling, "size": size, "align": align, "indirect": False, "union": False,
                    "leaves": [[0, leaf, part, element["align"]], [part, leaf, part, element["align"]]]}
        raise Unclassified(f"{spelling} (libclang type kind {kind})")

    def record(self, canonical, spelling, size, align):
        clang = self.clang
        declaration = clang.getTypeDeclaration(canonical)
        if clang.text(clang.getCursorSpelling(declaration)) == GUEST_LONG_DOUBLE:
            return {"kind": "x87", "type": "GuestLongDouble", "size": size, "align": align}
        if size < 0:
            raise Unclassified(f"{spelling} is incomplete")
        leaves = None
        if size <= FLATTENED_RECORD_BYTES:
            leaves = []
            self.flatten(canonical, 0, leaves)
            if any(leaf[1] == "long double" for leaf in leaves):
                raise Unclassified(f"{spelling} has a long double member, whose size differs between the guest and arm64")
            if not leaves:
                return {"kind": "empty", "type": spelling, "size": size, "align": align}
        return {"kind": "record", "type": spelling, "size": size, "align": align,
                "indirect": self.indirect(canonical), "union": declaration.kind == CURSOR_UNION, "leaves": leaves}

    def flatten(self, given, base, leaves):
        clang = self.clang
        canonical = clang.getCanonicalType(given)
        if canonical.kind == TYPE_RECORD:
            declaration = clang.getTypeDeclaration(canonical)
            if clang.text(clang.getCursorSpelling(declaration)) == GUEST_LONG_DOUBLE:
                leaves.append([base, "x87", 16, 16])
                return
            for child in clang.children(declaration):
                if child.kind == CURSOR_BASE:
                    offset = clang.getOffsetOfBase(declaration, child)
                    if clang.isVirtualBase(child) or offset < 0:
                        raise Unclassified("a virtual or unknown base class")
                    self.flatten(clang.getCursorType(child), base + offset // 8, leaves)
            for field in clang.fields(canonical):
                offset = clang.Cursor_getOffsetOfField(field)
                if offset < 0:
                    raise Unclassified(f"a field without an offset ({offset})")
                if clang.Cursor_isBitField(field):
                    width = clang.getFieldDeclBitWidth(field)
                    if width > 0:
                        leaves.append([base + offset // 8, "int", (offset % 8 + width + 7) // 8, 1])
                    continue
                self.flatten(clang.getCursorType(field), base + offset // 8, leaves)
            return
        if canonical.kind == TYPE_CONSTANT_ARRAY:
            element = clang.getElementType(canonical)
            step = clang.Type_getSizeOf(element)
            for index in range(clang.getArraySize(canonical)):
                self.flatten(element, base + index * step, leaves)
            return
        if canonical.kind == TYPE_INCOMPLETE_ARRAY:
            return
        value = self.value(canonical, CALLBACK_DEPTH)
        if value["kind"] not in LEAF_KINDS:
            raise Unclassified(f"a member of type {value.get('type')}")
        leaves.append([base, LEAF_KINDS[value["kind"]], value["size"], value["align"]])

    def indirect(self, record):
        """Whether C++ passes and returns the record type through a hidden reference: a copy constructor, move
        constructor or destructor is non-trivial, or every copy and move constructor is deleted. A template
        instantiation's special members are read from its template, its fields from the instantiation."""
        clang = self.clang
        declaration = clang.getTypeDeclaration(record)
        key = clang.text(clang.getTypeSpelling(record))
        if key in self.indirect_records:
            return self.indirect_records[key]
        self.indirect_records[key] = False
        template = clang.getSpecializedCursorTemplate(declaration)
        result, copies = False, []
        for child in clang.children(declaration if clang.Cursor_isNull(template) else template):
            if child.kind == CURSOR_DESTRUCTOR:
                if clang.CXXMethod_isVirtual(child) or not (clang.CXXMethod_isDefaulted(child) or clang.CXXMethod_isDeleted(child)):
                    result = True
            elif child.kind == CURSOR_CONSTRUCTOR and (clang.CXXConstructor_isCopyConstructor(child) or clang.CXXConstructor_isMoveConstructor(child)):
                deleted = bool(clang.CXXMethod_isDeleted(child))
                copies.append(deleted)
                if not deleted and not clang.CXXMethod_isDefaulted(child):
                    result = True
            elif child.kind == CURSOR_METHOD and clang.CXXMethod_isVirtual(child):
                result = True
            elif child.kind == CURSOR_BASE:
                result = result or bool(clang.isVirtualBase(child)) or self.indirect_type(clang.getCursorType(child))
        for field in clang.fields(record):
            result = result or self.indirect_type(clang.getCursorType(field))
        if copies and all(copies):
            result = True
        self.indirect_records[key] = result
        return result

    def indirect_type(self, given):
        clang = self.clang
        canonical = clang.getCanonicalType(given)
        while canonical.kind == TYPE_CONSTANT_ARRAY:
            canonical = clang.getCanonicalType(clang.getElementType(canonical))
        return canonical.kind == TYPE_RECORD and self.indirect(canonical)

    def inclusions(self, unit):
        clang = self.clang
        files = set()

        @INCLUSION_VISITOR
        def visit(file, stack, depth, data):
            name = clang.text(clang.getFileName(file))
            if name.startswith(self.root):
                files.add(name)

        clang.getInclusions(unit, visit, None)
        return sorted(files)


WORKER = {}


def start_worker(library, directory):
    WORKER["clang"] = LibClang(library)
    os.chdir(directory)


def parse_unit(job):
    """Parse one translation unit: its errors, exports and the repository files it includes."""
    clang = WORKER["clang"]
    encoded = [argument.encode() for argument in job["arguments"]]
    arguments = (ctypes.c_char_p * len(encoded))(*encoded)
    index = clang.createIndex(0, 0)
    unit = _P()
    try:
        status = clang.parseTranslationUnit2(index, job["file"].encode(), arguments, len(encoded), None, 0,
                                             0 if job["full"] else SKIP_FUNCTION_BODIES, ctypes.byref(unit))
        if status != 0 or not unit.value:
            return {"errors": [f"libclang cannot parse it (error {status})"], "exports": [], "dependencies": [job["file"]]}
        errors = []
        for number in range(clang.getNumDiagnostics(unit)):
            diagnostic = clang.getDiagnostic(unit, number)
            if clang.getDiagnosticSeverity(diagnostic) >= SEVERITY_ERROR:
                errors.append(clang.text(clang.formatDiagnostic(diagnostic, DISPLAY_SOURCE_LOCATION)))
            clang.disposeDiagnostic(diagnostic)
        reader = UnitReader(clang, job["root"], job["full"])
        return {"errors": errors, "exports": reader.exports(unit), "dependencies": reader.inclusions(unit)}
    finally:
        if unit.value:
            clang.disposeTranslationUnit(unit)
        clang.disposeIndex(index)


def run(command):
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()


def locate_libclang(compiler):
    if os.environ.get("APS5_LIBCLANG"):
        return Path(os.environ["APS5_LIBCLANG"])
    found = Path(shutil.which(compiler) or compiler).resolve()
    if found.parent == Path("/usr/bin"):
        found = Path(run(["xcrun", "--find", "clang"])).resolve()
    library = found.parent.parent / "lib" / "libclang.dylib"
    if not library.is_file():
        raise RuntimeError(f"no libclang.dylib beside {found}; set APS5_LIBCLANG to one")
    return library


def unit_arguments(entry, sdk):
    arguments = entry["arguments"] if "arguments" in entry else shlex.split(entry["command"])
    source = os.path.normpath(os.path.join(entry["directory"], entry["file"]))
    result, skip = [], False
    for argument in arguments[1:]:
        if skip:
            skip = False
        elif argument in ("-o", "-MF", "-MT", "-MQ"):
            skip = True
        elif argument in ("-c", "-MD", "-MMD") or os.path.normpath(os.path.join(entry["directory"], argument)) == source:
            continue
        else:
            result.append(argument)
    if sdk and not any(argument.startswith(("-isysroot", "--sysroot")) for argument in result):
        result += ["-isysroot", sdk]
    return result


def library_units(commands, root):
    """The translation units of prx library targets: file -> its compile command and the libraries built from it."""
    # core/fex/tests/prx holds test-only libraries (the bridge's ABI fixture library), built only with tests.
    prefixes = [str(Path(root) / "core" / "libs" / "prx") + os.sep, str(Path(root) / "core" / "fex" / "tests" / "prx") + os.sep]
    units = {}
    for entry in commands:
        source = os.path.normpath(os.path.join(entry["directory"], entry["file"]))
        command = " ".join(entry["arguments"]) if "arguments" in entry else entry["command"]
        target = re.search(r"CMakeFiles/([^/]+)\.dir/", entry.get("output", "")) or re.search(r"-o\s+\S*CMakeFiles/([^/]+)\.dir/", command)
        prx = next((prefix for prefix in prefixes if source.startswith(prefix)), None)
        if prx is None or "tests" in Path(source[len(prx):]).parts:
            continue
        if not re.search(r"-D\S+_EXPORTS\b", command) or target is None:
            continue
        unit = units.setdefault(source, {"entry": entry, "libraries": set()})
        unit["libraries"].add(target.group(1))
    return units


def fingerprint(paths):
    result = []
    for path in paths:
        try:
            status = os.stat(path)
            result.append([path, status.st_mtime_ns, status.st_size])
        except OSError:
            result.append([path, 0, -1])
    return result


def parse_all(units, root, cache, jobs, library, full):
    """Parse the units that changed since the cache was written; return file -> parse result."""
    source = Path(__file__).read_text()
    generator = hashlib.sha256(source[:source.index("\ndef align_to(")].encode()).hexdigest()
    library_status = os.stat(library)
    directories = {unit["entry"]["directory"] for unit in units.values()}
    if len(directories) > 1:
        raise RuntimeError(f"compile_commands.json uses several directories: {sorted(directories)}")
    sdk = run(["xcrun", "--show-sdk-path"]) if sys.platform == "darwin" else ""
    results, pending = {}, []
    cache.mkdir(parents=True, exist_ok=True)
    for file, unit in sorted(units.items()):
        arguments = unit_arguments(unit["entry"], sdk)
        key = hashlib.sha256(json.dumps([generator, str(library), library_status.st_mtime_ns, root, full, arguments]).encode()).hexdigest()
        stored = cache / (hashlib.sha256(file.encode()).hexdigest()[:24] + ".json")
        try:
            cached = json.loads(stored.read_text())
        except (OSError, ValueError):
            cached = None
        if cached and cached["key"] == key and fingerprint(path for path, _, _ in cached["dependencies"]) == cached["dependencies"]:
            results[file] = cached["result"]
        else:
            pending.append(({"file": file, "arguments": arguments, "root": root, "full": full}, key, stored))
    if pending:
        with ProcessPoolExecutor(max_workers=jobs, initializer=start_worker, initargs=(str(library), directories.pop())) as pool:
            for (job, key, stored), result in zip(pending, pool.map(parse_unit, [item[0] for item in pending])):
                results[job["file"]] = result
                if not result["errors"]:
                    stored.write_text(json.dumps({"key": key, "dependencies": fingerprint(result["dependencies"]), "result": result}))
    return results, len(pending)


def align_to(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def guest_size(value):
    return 16 if value["kind"] in ("long double", "x87") else value.get("size", 8)


def guest_align(value):
    return 16 if value["kind"] in ("long double", "x87") else value.get("align", 8)


def merge_class(first, second):
    if first == second or second == "NONE":
        return first
    if first == "NONE":
        return second
    if "MEMORY" in (first, second) or {first, second} & {"X87", "X87UP"}:
        return "MEMORY"
    if "INTEGER" in (first, second):
        return "INTEGER"
    return "SSE"


def guest_classes(value):
    """The SysV eightbyte classes of a value, or MEMORY, INDIRECT, X87 or IGNORE."""
    kind = value["kind"]
    if kind in ("int", "pointer", "function"):
        return ["INTEGER"]
    if kind == "int128":
        return ["INTEGER", "INTEGER"]
    if kind in ("float", "double"):
        return ["SSE"]
    if kind in ("long double", "x87"):
        return "X87"
    if kind == "empty":
        return "IGNORE"
    if kind == "vector":
        return {8: ["SSE"], 16: ["SSE", "SSEUP"]}.get(value["size"], "MEMORY")
    if value["indirect"]:
        return "INDIRECT"
    if value["size"] > 16 or value["leaves"] is None:
        return "MEMORY"
    if [leaf[1] for leaf in value["leaves"]] == ["x87"]:
        return "X87"
    classes = ["NONE"] * ((value["size"] + 7) // 8)
    for offset, leaf, size, alignment in value["leaves"]:
        if offset % alignment or leaf == "x87":
            return "MEMORY"
        first = offset // 8
        if leaf == "vector" and size == 16:
            classes[first] = merge_class(classes[first], "SSE")
            classes[first + 1] = merge_class(classes[first + 1], "SSEUP")
            continue
        for eightbyte in range(first, (offset + size - 1) // 8 + 1):
            classes[eightbyte] = merge_class(classes[eightbyte], "SSE" if leaf in ("float", "double", "vector") else "INTEGER")
    return "MEMORY" if "MEMORY" in classes else classes


def guest_result(value):
    if value["kind"] == "void":
        return [], False
    classes = guest_classes(value)
    if classes == "IGNORE":
        return [], False
    if classes in ("INDIRECT", "MEMORY"):
        return [["sret", 0, guest_size(value), "reference"]], True
    if classes == "X87":
        return [["st0", 0, 16, "value"]], False
    pieces, integers, vectors, last = [], ["rax", "rdx"], ["xmm0", "xmm1"], None
    for index, kind in enumerate(classes):
        start = 8 * index
        length = min(8, guest_size(value) - start)
        if kind == "INTEGER":
            pieces.append([integers.pop(0), start, length, "value"])
        elif kind == "SSE":
            last = vectors.pop(0)
            pieces.append([last + "+0", start, length, "value"])
        elif kind == "SSEUP":
            pieces.append([last + "+8", start, length, "value"])
    return pieces, False


def guest_layout(signature):
    """Where the guest (x86-64 SysV) puts each parameter, and where it finds the result."""
    result, hidden = guest_result(signature["result"])
    integer, vector, stack = (1 if hidden else 0), 0, 0
    parameters = []
    for value in signature["parameters"]:
        classes = guest_classes(value)
        how = "reference" if classes == "INDIRECT" else "value"
        size = 8 if how == "reference" else guest_size(value)
        if classes == "IGNORE":
            parameters.append([])
            continue
        if classes == "INDIRECT":
            classes = ["INTEGER"]
        if isinstance(classes, list) and integer + classes.count("INTEGER") <= 6 and vector + classes.count("SSE") <= 8:
            pieces, last = [], None
            for index, kind in enumerate(classes):
                start = 8 * index
                length = min(8, size - start)
                if kind == "INTEGER":
                    pieces.append([GUEST_REGISTERS[integer], start, length, how])
                    integer += 1
                elif kind == "SSE":
                    last = f"xmm{vector}"
                    pieces.append([last + "+0", start, length, how])
                    vector += 1
                elif kind == "SSEUP":
                    pieces.append([last + "+8", start, length, how])
            parameters.append(pieces)
            continue
        stack = align_to(stack, 8 if how == "reference" else max(8, guest_align(value)))
        parameters.append([[f"stack+{stack}", 0, size, how]])
        stack += align_to(size, 8)
    return {"result": result, "hidden": hidden, "parameters": parameters}


def host_hfa(value):
    """(member size, member count) when arm64 passes the value as a homogeneous floating-point aggregate."""
    if value["kind"] != "record" or value["indirect"] or not value["leaves"]:
        return None
    shapes = {(leaf, size) for _, leaf, size, _ in value["leaves"]}
    if len(shapes) != 1:
        return None
    leaf, size = shapes.pop()
    count = len(value["leaves"])
    if leaf not in ("float", "double", "vector") or count > 4 or count * size != value["size"]:
        return None
    return size, count


def host_result(value):
    kind, size = value["kind"], value.get("size", 8)
    if kind in ("void", "empty"):
        return [], False
    if kind in ("int", "pointer", "function"):
        return [["x0", 0, size, "value"]], False
    if kind in ("float", "double", "long double", "vector"):
        return [["v0+0", 0, size, "value"]], False
    hfa = host_hfa(value)
    if hfa is not None:
        return [[f"v{index}+0", hfa[0] * index, hfa[0], "value"] for index in range(hfa[1])], False
    if kind == "record" and (value["indirect"] or size > 16):
        return [["sret", 0, size, "reference"]], True
    return [[f"x{index}", 8 * index, min(8, size - 8 * index), "value"] for index in range((size + 7) // 8)], False


def host_layout(signature):
    """Where Apple's arm64 convention puts each fixed parameter, and where it returns the result."""
    result, hidden = host_result(signature["result"])
    integer = vector = stack = 0
    parameters = []
    for value in signature["parameters"]:
        kind, size, how = value["kind"], value.get("size", 8), "value"
        hfa = host_hfa(value)
        if kind == "empty":
            parameters.append([])
            continue
        if kind == "record" and (value["indirect"] or (size > 16 and hfa is None)):
            kind, size, how = "pointer", 8, "reference"
        if kind in ("int", "pointer", "function"):
            if integer < 8:
                parameters.append([[f"x{integer}", 0, size, how]])
                integer += 1
            else:
                stack = align_to(stack, size)
                parameters.append([[f"stack+{stack}", 0, size, how]])
                stack += size
        elif kind in ("float", "double", "long double", "vector"):
            if vector < 8:
                parameters.append([[f"v{vector}+0", 0, size, how]])
                vector += 1
            else:
                stack = align_to(stack, size)
                parameters.append([[f"stack+{stack}", 0, size, how]])
                stack += size
        elif hfa is not None:
            member, count = hfa
            if vector + count <= 8:
                parameters.append([[f"v{vector + index}+0", member * index, member, how] for index in range(count)])
                vector += count
            else:
                vector = 8
                stack = align_to(stack, max(8, value["align"]))
                parameters.append([[f"stack+{stack}", 0, size, how]])
                stack += align_to(size, 8)
        else:
            words = (size + 7) // 8
            # Apple's convention does not start a 16-byte aligned value (__int128, an aligned struct) at an
            # even register, unlike AAPCS64 (checked with clang -target arm64-apple-macos14 and by verify).
            if integer + words <= 8:
                parameters.append([[f"x{integer + index}", 8 * index, min(8, size - 8 * index), how] for index in range(words)])
                integer += words
            else:
                integer = 8
                stack = align_to(stack, max(8, value.get("align", 8)))
                parameters.append([[f"stack+{stack}", 0, size, how]])
                stack += 8 * words
    return {"result": result, "hidden": hidden, "parameters": parameters}


def bridge_byte(location, index):
    """The arm64 byte the bridge's positional mapping fills from byte index of a guest location."""
    if location in GUEST_REGISTERS:
        return f"x{GUEST_REGISTERS.index(location)}", index
    if location.startswith("xmm"):
        register, base = location.split("+")
        return "v" + register[3:], int(base) + index
    if location.startswith("stack+"):
        at = int(location[6:]) + index
        if at < 16:
            return f"x{6 + at // 8}", at % 8
        return ("stack", at - 16) if at < POSITIONAL_STACK_BYTES else ("lost", at)
    return location, index


def result_byte(location, index):
    """The arm64 result byte the bridge copies into byte index of a guest result location."""
    register, _, base = location.partition("+")
    if register in GUEST_RESULT_REGISTERS:
        return GUEST_RESULT_REGISTERS[register], int(base or 0) + index
    return location, index


def host_byte(location, index):
    if location.startswith("v"):
        register, base = location.split("+")
        return register, int(base) + index
    if location.startswith("stack+"):
        return "stack", int(location[6:]) + index
    return location, index


def byte_map(pieces, place):
    return {(how, offset + index): place(location, index) for location, offset, size, how in pieces for index in range(size)}


def mismatch_reason(value, guest, host):
    kind = value["kind"]
    if kind in ("long double", "int128"):
        return {"long double": "long-double", "int128": "int128"}[kind]
    if kind == "x87":
        return "x87-argument"
    if kind == "record":
        if host and host[0][3] == "reference" and not (guest and guest[0][3] == "reference"):
            return "struct-memory"
        if any(not piece[0].startswith("stack+") for piece in guest + host):
            return "struct-registers"
    if any(piece[0].startswith("stack+") for piece in guest + host):
        return "stack-layout"
    return "register-shift"


def classify(signature):
    """Place the signature under both conventions and list why positional marshalling is wrong for it."""
    values = [signature["result"], *signature["parameters"]]
    if any(value["kind"] == "unsupported" for value in values):
        return None, None, ["unclassified"]
    guest, host = guest_layout(signature), host_layout(signature)
    reasons = set()
    if signature["variadic"]:
        reasons.add("variadic")
    if guest["hidden"] or host["hidden"]:
        reasons.add("sret")
    elif byte_map(guest["result"], result_byte) != byte_map(host["result"], host_byte):
        kind = signature["result"]["kind"]
        reasons.add({"x87": "x87-result", "long double": "long-double", "int128": "int128"}.get(kind, "struct-result"))
    for value, guest_pieces, host_pieces in zip(signature["parameters"], guest["parameters"], host["parameters"]):
        if byte_map(guest_pieces, bridge_byte) != byte_map(host_pieces, host_byte):
            reasons.add(mismatch_reason(value, guest_pieces, host_pieces))
    return guest, host, sorted(reasons)


def classify_callbacks(signature):
    for value in [signature["result"], *signature["parameters"]]:
        if value.get("callback") is not None:
            classify_callbacks(value["callback"])
            value["callback"]["reasons"] = classify(value["callback"])[2]


def untyped(value):
    """A value or signature without its C type names, which the calling conventions do not see."""
    if isinstance(value, dict):
        return {key: untyped(item) for key, item in value.items() if key != "type"}
    if isinstance(value, list):
        return [untyped(item) for item in value]
    return value


def merge(units, results, root):
    """Combine the per-unit exports: name -> export, plus the problems found."""
    exports, problems, host_only = {}, [], set()
    for file in sorted(results):
        for error in results[file]["errors"]:
            problems.append(f"{os.path.relpath(file, root)}: {error}")
        for found in results[file]["exports"]:
            if found["name"].endswith(HOST_ONLY):
                host_only.add(found["name"])
                continue
            signature = {key: found[key] for key in ("variadic", "result", "parameters")}
            export = exports.setdefault(found["name"], {"name": found["name"], "libraries": set(), "definitions": {}})
            export["libraries"].update(units[file]["libraries"])
            export["definitions"].setdefault(json.dumps(untyped(signature), sort_keys=True), (signature, found["location"], file))
    result = []
    for name, export in sorted(exports.items()):
        definitions = sorted(export["definitions"].values(), key=lambda item: item[1])
        if len(definitions) > 1:
            problems.append(f"{name}: definitions differ: " + ", ".join(location for _, location, _ in definitions))
        signature, location, unit = definitions[0]
        guest, host, reasons = classify(signature)
        if guest is None:
            problems.append(f"{location}: {name}: cannot classify {signature['result']['reason']}")
        else:
            classify_callbacks(signature)
        result.append({"name": name, "location": location, "unit": os.path.relpath(unit, root),
                       "libraries": sorted(export["libraries"]), **signature, "reasons": reasons,
                       "guest": guest, "host": host})
    return result, problems, len(host_only)


def piece_row(location, offset, size, how, result):
    """A BridgePiece initialiser: where one part of a value lives under one convention."""
    if location == "sret":
        place, register, at = "Sret", 0, 0
    elif location == "st0":
        place, register, at = "X87", 0, 0
    elif location.startswith("stack+"):
        place, register, at = "Stack", 0, int(location[6:])
    elif "+" in location:
        register, at = location.split("+")
        place, register, at = "Vector", int(register.lstrip("xmv")), int(at)
    elif result and location in ("rax", "rdx"):
        place, register, at = "Gpr", ("rax", "rdx").index(location), 0
    elif location in GUEST_REGISTERS:
        place, register, at = "Gpr", GUEST_REGISTERS.index(location), 0
    elif location.startswith("x") and location[1:].isdigit():
        place, register, at = "Gpr", int(location[1:]), 0
    else:
        raise RuntimeError(f"no bridge place for {location}")
    reference = "true" if how == "reference" else "false"
    return f"{{BridgePlace::{place}, {register}, {at}, {offset}, {size}, {reference}}}"


class TableWriter:
    """Writes the C++ table that BridgeSignatures.cpp includes."""

    def __init__(self):
        self.leaves, self.values, self.callbacks = [], [], []
        self.pieces, self.layouts = [], []

    def pieces_of(self, pieces, result):
        first = len(self.pieces)
        self.pieces += [piece_row(*piece, result) for piece in pieces]
        return first, len(pieces)

    def layout(self, signature):
        """One BridgeLayout per value (result first): its pieces under SysV (guest) and under Apple arm64 (host)."""
        count = 1 + len(signature["parameters"])
        if any(value["kind"] == "unsupported" for value in [signature["result"], *signature["parameters"]]):
            return ["{0, 0, 0, 0}"] * count
        guest, host = guest_layout(signature), host_layout(signature)
        rows = []
        for index, (guest_pieces, host_pieces) in enumerate(zip([guest["result"], *guest["parameters"]],
                                                                  [host["result"], *host["parameters"]])):
            guest_first, guest_count = self.pieces_of(guest_pieces, index == 0)
            host_first, host_count = self.pieces_of(host_pieces, index == 0)
            rows.append(f"{{{guest_first}, {guest_count}, {host_first}, {host_count}}}")
        return rows

    def value(self, value):
        kind = value["kind"]
        if kind == "int":
            name = "Signed" if value["signed"] else "Unsigned"
        else:
            name = KIND_NAMES[kind]
        leaves = value.get("leaves") or []
        first = len(self.leaves)
        self.leaves += [f"{{{offset}, {size}, {align}, BridgeLeafKind::{LEAF_NAMES[leaf]}}}" for offset, leaf, size, align in leaves]
        callback = "NoBridgeCallback"
        if value.get("callback"):
            callback = str(self.signature("", value["callback"], self.callbacks))
        indirect = "true" if value.get("indirect") else "false"
        return f"{{BridgeKind::{name}, {indirect}, {value.get('align', 0)}, {value.get('size', 0)}, {first}, {len(leaves)}, {callback}}}"

    def signature(self, name, signature, table):
        values = [self.value(value) for value in [signature["result"], *signature["parameters"]]]
        first = len(self.values)
        self.values += values
        self.layouts += self.layout(signature)
        reasons = " | ".join(f"BridgeReason::{REASONS[reason][0]}" for reason in signature.get("reasons", [])) or "0"
        table.append(f'{{"{name}", {first}, {len(signature["parameters"])}, {"true" if signature["variadic"] else "false"}, {reasons}}}')
        return len(table) - 1

    def write(self, exports, path):
        signatures = []
        for export in exports:
            self.signature(export["name"], export, signatures)
        lines = []
        for type_name, name, rows in (("BridgeLeaf", "Leaves", self.leaves), ("BridgeValue", "Values", self.values),
                                      ("BridgeSignature", "Callbacks", self.callbacks), ("BridgeSignature", "Signatures", signatures),
                                      ("BridgePiece", "Pieces", self.pieces), ("BridgeLayout", "Layouts", self.layouts)):
            if rows:
                lines.append(f"constexpr std::array<{type_name}, {len(rows)}> {name} {{{{")
                lines += [f"    {row}," for row in rows]
                lines.append("}};")
            else:
                lines.append(f"constexpr std::array<{type_name}, 0> {name} {{}};")
        write_if_changed(path, "\n".join(lines) + "\n")


def write_if_changed(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    else:
        os.utime(path)


def generate(options):
    root = str(Path(options.root).resolve())
    commands = json.loads(Path(options.compile_commands).read_text())
    units = library_units(commands, root)
    if not units:
        raise RuntimeError(f"{options.compile_commands} has no prx library translation units")
    entry = next(iter(units.values()))["entry"]
    compiler = entry["arguments"][0] if "arguments" in entry else shlex.split(entry["command"])[0]
    library = locate_libclang(compiler)
    results, parsed = parse_all(units, root, Path(options.cache), options.jobs, library, options.full_parse)
    exports, problems, host_only = merge(units, results, root)
    document = {"generator": "core/fex/tools/gen_bridge_signatures.py", "libclang": str(library), "units": len(units),
                "hostOnly": host_only, "problems": problems, "exports": exports}
    write_if_changed(options.json, json.dumps(document, indent=1) + "\n")
    TableWriter().write(exports, options.inc)
    positional = sum(1 for export in exports if not export["reasons"])
    print(f"{len(exports)} exports from {len(units)} translation units ({parsed} parsed, {len(units) - parsed} cached): "
          f"{positional} positional, {len(exports) - positional} need more, {len(problems)} problems")
    return 0


def hand_kept(bridge):
    """Bridge.cpp's VariadicExports (name -> fixed arguments) and X87Results (names)."""
    variadic_block = re.search(r"VariadicExports\[\]\s*=\s*\{(.*?)\};", bridge, re.S)
    x87_block = re.search(r"X87Results\[\]\s*=\s*\{(.*?)\};", bridge, re.S)
    if variadic_block is None or x87_block is None:
        raise RuntimeError("Bridge.cpp has no VariadicExports or X87Results table")
    variadic = {name: int(fixed) for name, fixed in re.findall(r'\{"(\w+)",\s*\w+,\s*(\d+)\}', variadic_block.group(1))}
    return variadic, set(re.findall(r'"(\w+)"', x87_block.group(1)))


def check(options):
    document = json.loads(Path(options.json).read_text())
    variadic, x87 = hand_kept(Path(options.bridge).read_text())
    exports = {export["name"]: export for export in document["exports"]}
    failures = list(document["problems"])
    generated_variadic = {name: len(export["parameters"]) for name, export in exports.items() if export["variadic"]}
    for name in sorted(set(generated_variadic) - set(variadic)):
        failures.append(f"{exports[name]['location']}: {name} is variadic with {generated_variadic[name]} fixed arguments but not in VariadicExports")
    for name in sorted(set(variadic) - set(generated_variadic)):
        failures.append(f"VariadicExports lists {name}, which no library exports as variadic")
    for name in sorted(set(variadic) & set(generated_variadic)):
        if variadic[name] != generated_variadic[name]:
            failures.append(f"{exports[name]['location']}: {name} has {generated_variadic[name]} fixed arguments, VariadicExports says {variadic[name]}")
    generated_x87 = {name for name, export in exports.items() if export["result"]["kind"] == "x87"}
    for name in sorted(generated_x87 - x87):
        failures.append(f"{exports[name]['location']}: {name} returns GuestLongDouble but is not in X87Results")
    for name in sorted(x87 - generated_x87):
        failures.append(f"X87Results lists {name}, which no library exports as returning GuestLongDouble")
    groups = {}
    for name, export in exports.items():
        for reason in export["reasons"]:
            groups.setdefault(reason, []).append(export)
    for reason, (_, description) in REASONS.items():
        if reason in groups:
            print(f"{reason} ({len(groups[reason])}): {description}")
            for export in groups[reason]:
                if options.verbose or reason not in ("variadic",):
                    others = ", ".join(other for other in export["reasons"] if other != reason)
                    print(f"    {export['name']}  {export['location']}" + (f"  (also {others})" if others else ""))
    stacked = [export for export in exports.values() if export["host"] and
               any(piece[0].startswith("stack+") for pieces in export["host"]["parameters"] for piece in pieces)]
    print(f"stack arguments on arm64 ({len(stacked)}):")
    for export in stacked:
        print(f"    {export['name']}  {len(export['parameters'])} arguments, {', '.join(export['reasons']) or 'positional'}")
    callbacks = [(export, value) for export in exports.values() for value in export["parameters"] if value.get("callback")]
    returned = [export["name"] for export in exports.values() if export["result"].get("callback")]
    print(f"function pointer arguments: {len(callbacks)} in {len({export['name'] for export, _ in callbacks})} exports, "
          f"{sum(1 for _, value in callbacks if value['callback']['reasons'])} with a callback that is not positional; "
          f"function pointer results: {', '.join(returned) or 'none'}")
    positional = sum(1 for export in exports.values() if not export["reasons"])
    print(f"{len(exports)} exports: {positional} positional, {len(exports) - positional} need more than positional marshalling; "
          f"VariadicExports {len(variadic)}, X87Results {len(x87)}")
    for failure in failures:
        print(f"error: {failure}")
    return 1 if failures else 0


PROBE_TARGETS = {"guest": "x86_64-sie-ps5", "host": "arm64-apple-macos14"}
PROBE_INTEGERS = {(1, False): "unsigned char", (1, True): "signed char", (2, False): "unsigned short", (2, True): "short",
                  (4, False): "unsigned int", (4, True): "int", (8, False): "unsigned long long", (8, True): "long long",
                  (16, False): "unsigned __int128", (16, True): "__int128"}
PROBE_LEAVES = {"float": "float", "double": "double", "long double": "long double"}
X86_REGISTERS = {name: canonical for canonical, names in {
    "rdi": "rdi edi di dil", "rsi": "rsi esi si sil", "rdx": "rdx edx dx dl", "rcx": "rcx ecx cx cl",
    "r8": "r8 r8d r8w r8b", "r9": "r9 r9d r9w r9b", "rax": "rax eax ax al", "st0": "fp0"}.items() for name in names.split()}


class ProbeWriter:
    """C++ functions whose parameters and result have the shapes of the exports, for one target."""

    def __init__(self, target):
        self.target = target
        self.definitions = []
        self.names = {}

    def named(self, key, make):
        if key not in self.names:
            self.names[key] = f"Shape{len(self.names)}"
            self.definitions.append(make(self.names[key]))
        return self.names[key]

    def leaf(self, leaf, size):
        if leaf == "int":
            return PROBE_INTEGERS.get((size, False), "unsigned char"), "" if (size, False) in PROBE_INTEGERS else f"[{size}]"
        if leaf == "x87":
            return self.type({"kind": "x87"}), ""
        if leaf == "vector":
            return self.named(("vector", size), lambda name: f"typedef float {name} __attribute__((vector_size({size})));"), ""
        return PROBE_LEAVES[leaf], ""

    def record(self, value):
        def make(name):
            members, at = [], 0
            for index, (offset, leaf, size, _) in enumerate(sorted(value["leaves"] or [])):
                if offset < at:
                    raise Unclassified("overlapping members")
                if offset > at:
                    members.append(f"unsigned char pad{index}[{offset - at}];")
                spelled, suffix = self.leaf(leaf, size)
                members.append(f"{spelled} m{index}{suffix};")
                at = offset + size
            if at < value["size"]:
                members.append(f"unsigned char tail[{value['size'] - at}];")
            if value["indirect"]:
                members.append(f"{name}(); {name}(const {name}&); ~{name}();")
            packed = any(offset % align for offset, _, _, align in value["leaves"] or [])
            attribute = " __attribute__((packed))" if packed else ""
            return f"struct alignas({value['align']}){attribute} {name} {{ {' '.join(members)} }};"
        if value.get("union"):
            raise Unclassified("a union")
        return self.named(json.dumps(untyped(value), sort_keys=True), make)

    def type(self, value):
        kind = value["kind"]
        if kind == "int":
            return PROBE_INTEGERS[(value["size"], value["signed"])]
        if kind == "int128":
            return PROBE_INTEGERS[(16, value["signed"])]
        if kind in ("void", "float", "double", "long double"):
            return kind
        if kind == "pointer":
            return "void*"
        if kind == "function":
            return self.named("function", lambda name: f"typedef void (*{name})();")
        if kind == "x87":
            if self.target == "guest":
                return "long double"
            return self.named("x87", lambda name: f"struct alignas(16) {name} {{ unsigned long long significand; unsigned short signExponent; }};")
        if kind == "vector":
            return self.leaf("vector", value["size"])[0]
        if kind == "empty":
            return self.named(("empty", value["size"]), lambda name: f"struct {name} {{}};")
        if kind == "record":
            return self.record(value)
        raise Unclassified(f"a {kind} value")

    def function(self, name, signature):
        parameters = [f"{self.type(value)} p{index}" for index, value in enumerate(signature["parameters"])]
        if signature["variadic"]:
            parameters.append("...")
        result = self.type(signature["result"])
        body = "" if result == "void" else "return {};"
        return f'extern "C" {result} {name}({", ".join(parameters)}) {{ {body} }}'


def mir_functions(text, target):
    """name -> (argument registers in order, argument stack ranges, result registers) from MIR after instruction
    selection; the result registers are None for a function without a return (it calls a noreturn function such as
    NotImplemented_nid_no_patch), whose result locations the compiler never shows."""
    functions = {}
    for document in re.split(r"^---\s*$", text, flags=re.M):
        name = re.search(r"^name:\s+(\S+)", document, re.M)
        if name is None:
            continue
        registers = re.search(r"^liveins:\s*\n((?:\s+- \{ reg:.*\n)*)", document, re.M)
        incoming = [normalize_register(register, target) for register in re.findall(r"reg: '\$(\w+)'", registers.group(1) if registers else "")]
        fixed = document.split("\nfixedStack:", 1)[1].split("\nstack:", 1)[0] if "\nfixedStack:" in document else ""
        stack = [(int(offset), int(size)) for offset, size in re.findall(r"- \{ id: \d+, type: \w+, offset: (-?\d+), size: (\d+)", fixed)]
        classes = dict(re.findall(r"- \{ id: (\d+), class: (\w+)", document))
        returns = re.findall(r"^\s*RET\S*.*$", document, re.M)
        returned = [] if returns else None
        for line in returns:
            for register, virtual in re.findall(r"\$(\w+)|%(\d+)", line):
                if register or classes.get(virtual, "").startswith("rfp"):
                    returned.append(normalize_register(register, target) if register else "st0")
        functions[name.group(1)] = (incoming, merge_stack(stack, target), None if returned is None else list(dict.fromkeys(returned)))
    return functions


def normalize_register(register, target):
    if target == "host":
        match = re.fullmatch(r"([wxsdqhb])(\d+)", register)
        return ("x" if match.group(1) in "wx" else "v") + match.group(2) if match else register
    return X86_REGISTERS.get(register, register)


def merge_stack(ranges, target):
    """Sorted stack ranges with touching ones joined, so an argument split into parts compares equal to one passed
    whole; guest ranges are whole 8-byte slots."""
    merged = []
    for offset, size in sorted(ranges):
        if target == "guest":
            size = align_to(size, 8)
        if merged and merged[-1][0] + merged[-1][1] == offset:
            merged[-1] = (merged[-1][0], merged[-1][1] + size)
        else:
            merged.append((offset, size))
    return merged


def model_locations(layout, target):
    """A layout's argument registers in order, argument stack ranges and result registers, as mir_functions
    reports them."""
    incoming = [("x8" if target == "host" else "rdi")] if layout["hidden"] else []
    stack = []
    for pieces in layout["parameters"]:
        for location, offset, size, how in pieces:
            if location.startswith("stack+"):
                stack.append((int(location[6:]), size))
            elif not incoming or incoming[-1] != location.split("+")[0]:
                incoming.append(location.split("+")[0])
    if layout["hidden"]:
        returned = ["rax"] if target == "guest" else []
    else:
        returned = list(dict.fromkeys(location.split("+")[0] for location, _, _, _ in layout["result"]))
    return incoming, merge_stack(stack, target), returned


def compiled_exports(document, commands_path, root, work):
    """Compile, as the build does but unoptimised, every unit that defines an export passing a struct, GuestLongDouble,
    __int128 or an empty class by value, and compare arm64's locations for those exports with the generator's."""
    units = {}
    for export in document["exports"]:
        values = [export["result"], *export["parameters"]]
        if export["host"] is not None and any(value["kind"] in ("record", "x87", "int128", "empty") for value in values):
            units.setdefault(export["unit"], []).append(export)
    entries = {}
    for entry in json.loads(Path(commands_path).read_text()):
        entries.setdefault(os.path.normpath(os.path.join(entry["directory"], entry["file"])), entry)
    sdk = run(["xcrun", "--show-sdk-path"])

    def compile_unit(item):
        index, (unit, exports) = item
        source = os.path.join(root, unit)
        entry = entries[source]
        compiler = entry["arguments"][0] if "arguments" in entry else shlex.split(entry["command"])[0]
        output = work / f"unit{index}.mir"
        subprocess.run([compiler, *unit_arguments(entry, sdk), "-O0", "-w", "-S", "-mllvm", "-global-isel=false", "-mllvm",
                        "-stop-after=finalize-isel", source, "-o", str(output)], cwd=entry["directory"], check=True, capture_output=True)
        compiled = mir_functions(output.read_text(), "host")
        output.unlink()
        return [(export, compiled.get(export["name"])) for export in exports]

    failures, compared = [], 0
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        for results in pool.map(compile_unit, enumerate(sorted(units.items()))):
            for export, found in results:
                expected = model_locations(host_layout(export), "host")
                compared += 1
                if found is not None and found[2] is None:
                    expected = (*expected[:2], None)
                if found != expected:
                    failures.append(f"host {export['name']} ({export['location']}): generator {expected}, compiler {found}")
    return compared, failures


def verify(options):
    """Compile every distinct export and callback shape for both targets and compare the compiler's argument and
    result locations with the generator's; with --compile-commands, also the library units whose exports pass
    structs by value."""
    document = json.loads(Path(options.json).read_text())
    shapes = {}

    def collect(signature):
        if any(value["kind"] == "unsupported" for value in [signature["result"], *signature["parameters"]]):
            return
        shapes.setdefault(json.dumps(untyped({key: signature[key] for key in ("variadic", "result", "parameters")}), sort_keys=True), signature)
        for value in [signature["result"], *signature["parameters"]]:
            if value.get("callback"):
                collect(value["callback"])

    for export in document["exports"]:
        collect(export)
    work = Path(options.work)
    work.mkdir(parents=True, exist_ok=True)
    failures, unverified, compared = [], [], 0
    layouts = {"guest": guest_layout, "host": host_layout}
    for target, triple in PROBE_TARGETS.items():
        writer, functions, skipped = ProbeWriter(target), [], set()
        for index, signature in enumerate(shapes.values()):
            try:
                functions.append(writer.function(f"probe{index}", signature))
            except Unclassified as error:
                skipped.add(index)
                unverified.append(f"{target} probe{index}: {error}")
        source = work / f"probes.{target}.cpp"
        source.write_text("\n".join(writer.definitions + functions) + "\n")
        output = work / f"probes.{target}.mir"
        subprocess.run(["xcrun", "clang++", "-std=c++20", "-O0", "-S", "-w", "-target", triple, "-mllvm", "-global-isel=false",
                        "-mllvm", "-stop-after=finalize-isel", str(source), "-o", str(output)], check=True)
        compiled = mir_functions(output.read_text(), target)
        for index, signature in enumerate(shapes.values()):
            if index in skipped:
                continue
            expected = model_locations(layouts[target](signature), target)
            compared += 1
            if expected != compiled[f"probe{index}"]:
                failures.append(f"{target} probe{index}: generator {expected}, compiler {compiled[f'probe{index}']} ({source})")
    units = 0
    if options.compile_commands:
        units, unit_failures = compiled_exports(document, options.compile_commands, str(Path(options.root).resolve()), work)
        failures += unit_failures
    for line in unverified:
        print(f"unverified: {line}")
    for line in failures:
        print(f"error: {line}")
    print(f"{len(shapes)} shapes, {compared} target layouts and {units} exports compiled from library units compared, "
          f"{len(unverified)} unverified, {len(failures)} differ")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    generating = commands.add_parser("generate")
    generating.add_argument("--compile-commands", required=True)
    generating.add_argument("--root", required=True)
    generating.add_argument("--cache", required=True)
    generating.add_argument("--json", required=True)
    generating.add_argument("--inc", required=True)
    generating.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    generating.add_argument("--full-parse", action="store_true", help="parse function bodies too and ask libclang which declarations define")
    checking = commands.add_parser("check")
    checking.add_argument("--json", required=True)
    checking.add_argument("--bridge", required=True)
    checking.add_argument("--verbose", action="store_true")
    verifying = commands.add_parser("verify")
    verifying.add_argument("--json", required=True)
    verifying.add_argument("--work", required=True)
    verifying.add_argument("--compile-commands")
    verifying.add_argument("--root", default=".")
    options = parser.parse_args()
    return {"generate": generate, "check": check, "verify": verify}[options.command](options)


if __name__ == "__main__":
    sys.exit(main())
