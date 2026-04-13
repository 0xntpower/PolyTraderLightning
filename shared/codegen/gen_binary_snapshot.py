"""Generate ``binary_snapshot.py`` and ``BinarySnapshot.hpp`` from the
schema TOML.

Runs as a build step from:
  * The PolyLiveVisualizer MSBuild pre-build event.
  * The PolyTraderLightning pytest conftest (indirectly, via import).
  * Manual invocation: ``python shared/codegen/gen_binary_snapshot.py``.

Idempotency: outputs are byte-compared with their existing content and
only rewritten when they differ. MSBuild-visible file mtimes stay stable,
so re-running the generator does not trigger a full C++ rebuild.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import tomllib
from pathlib import Path
from typing import Final, Literal, TypedDict

# ---------------------------------------------------------------------------
# Paths — derived from the generator's own location so the script works
# regardless of the current working directory.
# ---------------------------------------------------------------------------

_HERE: Final = Path(__file__).resolve().parent
_SHARED_DIR: Final = _HERE.parent
_PTL_ROOT: Final = _SHARED_DIR.parent
_REPO_ROOT: Final = _PTL_ROOT.parent

SCHEMA_PATH: Final = _SHARED_DIR / "binary_snapshot_schema.toml"
PY_OUT_PATH: Final = _SHARED_DIR / "binary_snapshot.py"
CPP_OUT_PATH: Final = (
    _REPO_ROOT / "PolyLiveVisualizer" / "src" / "PolyLiveVisualizer" / "data" / "BinarySnapshot.hpp"
)
GOLDEN_BIN_PATH: Final = _HERE / "golden_snapshot.bin"
GOLDEN_JSON_PATH: Final = _HERE / "golden_snapshot.json"


# ---------------------------------------------------------------------------
# Type mapping — schema type strings → (struct format char, C++ type, byte
# width, python category).
# ---------------------------------------------------------------------------

PyCategory = Literal["int", "float", "bool_u8"]


class TypeInfo(TypedDict):
    struct_char: str
    cpp_type: str
    size: int
    py_category: PyCategory


_TYPE_MAP: Final[dict[str, TypeInfo]] = {
    "u8": {"struct_char": "B", "cpp_type": "std::uint8_t", "size": 1, "py_category": "int"},
    "i8": {"struct_char": "b", "cpp_type": "std::int8_t", "size": 1, "py_category": "int"},
    "u16": {"struct_char": "H", "cpp_type": "std::uint16_t", "size": 2, "py_category": "int"},
    "i16": {"struct_char": "h", "cpp_type": "std::int16_t", "size": 2, "py_category": "int"},
    "u32": {"struct_char": "I", "cpp_type": "std::uint32_t", "size": 4, "py_category": "int"},
    "i32": {"struct_char": "i", "cpp_type": "std::int32_t", "size": 4, "py_category": "int"},
    "u64": {"struct_char": "Q", "cpp_type": "std::uint64_t", "size": 8, "py_category": "int"},
    "i64": {"struct_char": "q", "cpp_type": "std::int64_t", "size": 8, "py_category": "int"},
    "f32": {"struct_char": "f", "cpp_type": "float", "size": 4, "py_category": "float"},
    "f64": {"struct_char": "d", "cpp_type": "double", "size": 8, "py_category": "float"},
}


# ---------------------------------------------------------------------------
# Schema parsing
# ---------------------------------------------------------------------------


class FieldSpec(TypedDict):
    name: str
    type: str
    group: str
    doc: str


class Schema(TypedDict):
    version: int
    endianness: str
    fields: list[FieldSpec]
    struct_format: str
    total_size: int


def load_schema() -> Schema:
    raw = tomllib.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    version_any = raw.get("schema_version")
    if not isinstance(version_any, int) or version_any <= 0:
        raise ValueError("schema_version must be a positive int")
    endian = raw.get("endianness")
    if endian != "little":
        raise ValueError("endianness must be 'little'")

    fields_raw = raw.get("fields")
    if not isinstance(fields_raw, list) or not fields_raw:
        raise ValueError("fields must be a non-empty list")

    parsed: list[FieldSpec] = []
    seen_names: set[str] = set()
    for entry in fields_raw:
        if not isinstance(entry, dict):
            raise ValueError(f"field entry must be a table: {entry!r}")
        name = entry.get("name")
        type_str = entry.get("type")
        group = entry.get("group", "")
        doc = entry.get("doc", "")
        if not isinstance(name, str) or not name:
            raise ValueError(f"field missing name: {entry!r}")
        if not isinstance(type_str, str) or type_str not in _TYPE_MAP:
            raise ValueError(f"field {name!r}: unknown type {type_str!r}")
        if not isinstance(group, str):
            raise ValueError(f"field {name!r}: group must be string")
        if not isinstance(doc, str):
            raise ValueError(f"field {name!r}: doc must be string")
        if name in seen_names:
            raise ValueError(f"duplicate field name: {name!r}")
        seen_names.add(name)
        parsed.append(FieldSpec(name=name, type=type_str, group=group, doc=doc))

    struct_format = "<" + "".join(_TYPE_MAP[f["type"]]["struct_char"] for f in parsed)
    total_size = struct.calcsize(struct_format)

    return Schema(
        version=version_any,
        endianness=endian,
        fields=parsed,
        struct_format=struct_format,
        total_size=total_size,
    )


# ---------------------------------------------------------------------------
# Deterministic golden values — same rule used on both sides.
# ---------------------------------------------------------------------------


def golden_value(index: int, type_str: str) -> int | float:
    """Pick a deterministic value for field ``index`` of the given type.

    Int-shaped fields get ``index * 3 + 1`` (masked to type width) so every
    field has a distinct positive value. Float fields get ``index * 1.5 + 0.25``
    which keeps the value nonzero, non-integer, and distinct per index.
    """
    category = _TYPE_MAP[type_str]["py_category"]
    if category == "float":
        return float(index) * 1.5 + 0.25
    # int or bool_u8
    raw = index * 3 + 1
    size = _TYPE_MAP[type_str]["size"]
    is_signed = type_str in ("i8", "i16", "i32", "i64")
    if is_signed:
        bits = size * 8 - 1
        return raw % (1 << bits)
    bits = size * 8
    return raw % (1 << bits)


def build_golden(schema: Schema) -> tuple[bytes, dict[str, int | float]]:
    values: dict[str, int | float] = {}
    ordered: list[int | float] = []
    for i, f in enumerate(schema["fields"]):
        # Override the version byte so the golden always matches the live
        # schema version — this is the field parsers check first.
        if f["name"] == "version":
            v: int | float = schema["version"]
        else:
            v = golden_value(i, f["type"])
        values[f["name"]] = v
        ordered.append(v)
    packed = struct.pack(schema["struct_format"], *ordered)
    return packed, values


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------


_BANNER = (
    "# === GENERATED FILE — DO NOT EDIT ===\n"
    "# Regenerate via shared/codegen/gen_binary_snapshot.py.\n"
    "# Source schema: shared/binary_snapshot_schema.toml\n"
)

_BANNER_CPP = (
    "// === GENERATED FILE — DO NOT EDIT ===\n"
    "// Regenerate via shared/codegen/gen_binary_snapshot.py.\n"
    "// Source schema: PolyTraderLightning/shared/binary_snapshot_schema.toml\n"
)


def _schema_hash() -> str:
    raw = SCHEMA_PATH.read_bytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def emit_python(schema: Schema) -> str:
    fields = schema["fields"]
    schema_hash = _schema_hash()

    lines: list[str] = []
    lines.append('"""Binary snapshot wire format — generated from schema.\n\n')
    lines.append("Do not edit by hand. This module is the single source of truth\n")
    lines.append("for the struct layout used by ``state_publisher.publish_binary``\n")
    lines.append('and the matching C++ ``BinarySnapshot.hpp`` parser.\n"""\n')
    lines.append("\n")
    lines.append("from __future__ import annotations\n")
    lines.append("\n")
    lines.append("import struct\n")
    lines.append("from typing import Final, NamedTuple\n")
    lines.append("\n")
    lines.append(_BANNER)
    lines.append(f"# Schema SHA256 (first 16 hex): {schema_hash}\n")
    lines.append("\n")
    lines.append(f"SCHEMA_VERSION: Final[int] = {schema['version']}\n")
    lines.append(f'STRUCT_FORMAT: Final[str] = "{schema["struct_format"]}"\n')
    lines.append("STRUCT: Final[struct.Struct] = struct.Struct(STRUCT_FORMAT)\n")
    lines.append(f"SIZE: Final[int] = {schema['total_size']}\n")
    lines.append("if STRUCT.size != SIZE:  # pragma: no cover\n")
    lines.append(
        '    raise RuntimeError("binary_snapshot: struct.Struct size mismatch — regenerate")\n'
    )
    lines.append("\n")
    lines.append("FIELD_NAMES: Final[tuple[str, ...]] = (\n")
    lines.extend(f'    "{f["name"]}",\n' for f in fields)
    lines.append(")\n")
    lines.append("\n")
    lines.append("\n")
    lines.append("class HotSnapshot(NamedTuple):\n")
    lines.append('    """Strongly-typed hot-path field tuple.\n\n')
    lines.append("    Construction order matches the schema — the caller passes\n")
    lines.append("    fields by keyword so reordering the schema breaks loudly.\n")
    lines.append('    """\n')
    lines.append("\n")
    for f in fields:
        cat = _TYPE_MAP[f["type"]]["py_category"]
        py_type = "float" if cat == "float" else "int"
        lines.append(f"    {f['name']}: {py_type}\n")
    lines.append("\n")
    lines.append("\n")
    lines.append("def pack(snap: HotSnapshot) -> bytes:\n")
    lines.append('    """Serialize ``snap`` to the wire format. One allocation."""\n')
    lines.append("    return STRUCT.pack(*snap)\n")
    lines.append("\n")
    lines.append("\n")
    lines.append("def pack_into(buf: bytearray, offset: int, snap: HotSnapshot) -> None:\n")
    lines.append('    """Serialize ``snap`` directly into ``buf[offset:]`` — zero alloc."""\n')
    lines.append("    STRUCT.pack_into(buf, offset, *snap)\n")
    lines.append("\n")
    lines.append("\n")
    lines.append("def unpack(data: bytes | memoryview | bytearray) -> HotSnapshot:\n")
    lines.append('    """Parse wire bytes back into a ``HotSnapshot``. Used by tests."""\n')
    lines.append("    return HotSnapshot(*STRUCT.unpack(data))\n")
    return "".join(lines)


def emit_cpp(schema: Schema) -> str:
    fields = schema["fields"]
    schema_hash = _schema_hash()

    lines: list[str] = []
    lines.append("#pragma once\n")
    lines.append("\n")
    lines.append(_BANNER_CPP)
    lines.append(f"// Schema SHA256 (first 16 hex): {schema_hash}\n")
    lines.append("\n")
    lines.append("// BinarySnapshot.hpp — packed struct matching the publisher's\n")
    lines.append("// ``STRUCT_FORMAT`` in ``binary_snapshot.py``. Layout is little-\n")
    lines.append("// endian, byte-packed, and directly ``memcpy``-able because every\n")
    lines.append("// field is a trivially-copyable primitive.\n")
    lines.append("\n")
    lines.append("#include <cstddef>\n")
    lines.append("#include <cstdint>\n")
    lines.append("\n")
    lines.append("namespace plv::data {\n")
    lines.append("\n")
    lines.append(f"inline constexpr std::uint8_t kBinarySchemaVersion = {schema['version']};\n")
    lines.append("\n")
    lines.append("#pragma pack(push, 1)\n")
    lines.append("struct BinarySnapshot {\n")
    for f in fields:
        cpp_t = _TYPE_MAP[f["type"]]["cpp_type"]
        doc = f["doc"]
        suffix = f"  // {doc}" if doc else ""
        lines.append(f"    {cpp_t} {f['name']};{suffix}\n")
    lines.append("};\n")
    lines.append("#pragma pack(pop)\n")
    lines.append("\n")
    lines.append("inline constexpr std::size_t kBinarySnapshotBytes = sizeof(BinarySnapshot);\n")
    lines.append(
        f"static_assert(kBinarySnapshotBytes == {schema['total_size']}, "
        '"BinarySnapshot size drift — regenerate BinarySnapshot.hpp");\n'
    )
    lines.append(
        "static_assert(offsetof(BinarySnapshot, version) == 0, "
        '"version byte must be at offset 0");\n'
    )
    lines.append("\n")
    lines.append("} // namespace plv::data\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# Idempotent writer
# ---------------------------------------------------------------------------


def _write_if_changed(path: Path, content: str | bytes) -> bool:
    data = content.encode("utf-8") if isinstance(content, str) else content
    if path.exists() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    try:
        schema = load_schema()
    except (OSError, ValueError) as e:
        print(f"gen_binary_snapshot: schema error: {e}", file=sys.stderr)
        return 1

    py_src = emit_python(schema)
    cpp_src = emit_cpp(schema)
    golden_bytes, golden_values = build_golden(schema)

    # Emit the golden values as a human/C++-readable JSON sidecar so the
    # visualizer self-test can verify field-by-field without reimplementing
    # the "golden_value" formula.
    golden_json_str = (
        json.dumps(
            {
                "schema_version": schema["version"],
                "total_size": schema["total_size"],
                "struct_format": schema["struct_format"],
                "fields": [
                    {
                        "name": f["name"],
                        "type": f["type"],
                        "value": golden_values[f["name"]],
                    }
                    for f in schema["fields"]
                ],
            },
            indent=2,
        )
        + "\n"
    )

    changes: list[str] = []
    if _write_if_changed(PY_OUT_PATH, py_src):
        changes.append(str(PY_OUT_PATH.relative_to(_PTL_ROOT)))
    if _write_if_changed(CPP_OUT_PATH, cpp_src):
        try:
            rel_cpp = CPP_OUT_PATH.relative_to(_REPO_ROOT)
        except ValueError:
            rel_cpp = CPP_OUT_PATH
        changes.append(str(rel_cpp))
    if _write_if_changed(GOLDEN_BIN_PATH, golden_bytes):
        changes.append(str(GOLDEN_BIN_PATH.relative_to(_PTL_ROOT)))
    if _write_if_changed(GOLDEN_JSON_PATH, golden_json_str):
        changes.append(str(GOLDEN_JSON_PATH.relative_to(_PTL_ROOT)))

    if changes:
        print(f"gen_binary_snapshot: regenerated {len(changes)} file(s)")
        for c in changes:
            print(f"  {c}")
    else:
        print("gen_binary_snapshot: up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
