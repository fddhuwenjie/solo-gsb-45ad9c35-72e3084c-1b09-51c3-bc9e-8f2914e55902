"""HTTP request materialization for one OpenAPI operation (``POST /requests/materialize``).

Given an OpenAPI 3.x document, an operation selector (``"METHOD /path"``) and
a grouping of parameter values keyed by parameter location, this module
builds the concrete pieces of an HTTP request:

* ``request_target`` -- the RFC 7230 origin-form target (path plus query);
* ``headers`` -- every ``in: header`` parameter rendered per its style;
* ``cookie`` -- the ``Cookie`` header value produced from ``in: cookie``
  parameters, rendered with the ``form`` style.

Parameter lists are resolved cycle-safe for local ``$ref`` entries. The
path-level parameter list is merged with the operation-level list, and an
operation parameter with the same ``(in, name)`` pair overrides the
path-level one, as mandated by the OpenAPI specification.

Serialization follows the OpenAPI 3 parameter serialization table:

* path: ``simple`` (default), ``label``, ``matrix``;
* query: ``form`` (default), ``spaceDelimited``, ``pipeDelimited``,
  ``deepObject``;
* header: ``simple`` only;
* cookie: ``form`` only.

Default ``explode`` values are derived from the style when ``explode`` is not
declared. Primitive, array and object values are supported everywhere the
serialization table supports them; unsupported shape/style combinations and
unusable values are reported as 422-class issues with a JSON Pointer to the
parameter (in the document) and to the value (in the request).

Output ordering is stable and independent of input key order: query
parameters, headers and cookies are emitted by name, object members by key.
The input document and request values are never mutated.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote

from .refresolver import (
    ReferenceError,
    RefResolver,
    UnresolvableReferenceError,
    is_local_ref,
)
from .slicer import HTTP_METHODS, validate_openapi_shape

PARAM_LOCATIONS = ("path", "query", "header", "cookie")

# Default explode per (location, style); the OpenAPI 3 table:
# simple/label  -> false, matrix/form -> true, the named array/object query
# styles do not support explode=true and are rendered as explode=false
# (deepObject has its own object encoding).
DEFAULT_EXPLODE = {
    ("path", "simple"): False,
    ("path", "label"): False,
    ("path", "matrix"): True,
    ("query", "form"): True,
    ("query", "spaceDelimited"): False,
    ("query", "pipeDelimited"): False,
    ("query", "deepObject"): True,
    ("header", "simple"): False,
    ("cookie", "form"): False,
}

VALID_STYLES = {
    "path": {"simple", "label", "matrix"},
    "query": {"form", "spaceDelimited", "pipeDelimited", "deepObject"},
    "header": {"simple"},
    "cookie": {"form"},
}

# Token characters per RFC 7230 section 3.2.6 (VCHAR minus delimiters).
_TOKEN_CHARS = set(
    "!#$%&'*+-.^_`|~"
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)

_PATH_TEMPLATE = re.compile(r"\{([^{}]*)\}")


class MaterializeError(Exception):
    """Base class for materialization request failures."""


class InvalidMaterializeRequestError(MaterializeError):
    """The request itself is malformed (HTTP 400)."""

    def __init__(self, messages: List[str]) -> None:
        self.messages = messages
        super().__init__("; ".join(messages))


class MaterializationError(MaterializeError):
    """The contract/value combination cannot be rendered (HTTP 422)."""

    def __init__(self, issues: List[Dict[str, Any]]) -> None:
        self.issues = issues
        super().__init__(f"{len(issues)} materialization issue(s)")


# --------------------------------------------------------------------------- #
# JSON Pointer helpers
# --------------------------------------------------------------------------- #

def _escape_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _pointer(*tokens: Any) -> str:
    """Build a JSON Pointer from raw (unescaped) reference tokens."""
    return "".join("/" + _escape_token(str(token)) for token in tokens)


def _join_pointer(base: str, *tokens: Any) -> str:
    """Append raw reference tokens to an already-built pointer."""
    return base + _pointer(*tokens)

# --------------------------------------------------------------------------- #
# Selector handling
# --------------------------------------------------------------------------- #

def parse_selector(raw: Any) -> Optional[Tuple[str, str]]:
    """Parse one ``"METHOD /path"`` selector (same rule as ``/slice``)."""
    if not isinstance(raw, str):
        return None
    parts = raw.strip().split(None, 1)
    if len(parts) != 2:
        return None
    method, path = parts[0].lower(), parts[1]
    if method not in HTTP_METHODS or not path or not path.startswith("/"):
        return None
    return method, path


# --------------------------------------------------------------------------- #
# Issue collection
# --------------------------------------------------------------------------- #

def _issue(
    code: str,
    *,
    reason: str,
    location: Optional[str] = None,
    name: Optional[str] = None,
    pointer: Optional[str] = None,
    value_pointer: Optional[str] = None,
    ref: Optional[str] = None,
) -> Dict[str, Any]:
    item: Dict[str, Any] = {"code": code, "reason": reason}
    if location is not None:
        item["in"] = location
    if name is not None:
        item["name"] = name
    if pointer is not None:
        item["pointer"] = pointer
    if value_pointer is not None:
        item["value_pointer"] = value_pointer
    if ref is not None:
        item["ref"] = ref
    return item


def _sort_issues(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(item: Dict[str, Any]) -> Tuple[str, str, str, str, str, str]:
        return (
            item.get("code", ""),
            item.get("in", ""),
            item.get("name", ""),
            item.get("pointer", ""),
            item.get("value_pointer", ""),
            item.get("reason", ""),
        )

    return sorted(issues, key=key)


# --------------------------------------------------------------------------- #
# Path item / parameter resolution
# --------------------------------------------------------------------------- #

def _resolve_path_item(
    resolver: RefResolver,
    raw_item: Any,
    path_pointer: str,
    issues: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Effective path item, following a path-level ``$ref`` (3.1 siblings win).

    Returns ``None`` when the path item cannot be resolved or is not an
    object; an issue is recorded.
    """
    if not isinstance(raw_item, dict):
        issues.append(
            _issue(
                "invalid_path_item",
                reason="path item must be an object",
                pointer=path_pointer,
            )
        )
        return None
    ref = raw_item.get("$ref")
    if not isinstance(ref, str):
        return raw_item
    if not is_local_ref(ref):
        issues.append(
            _issue(
                "unsupported_reference",
                reason=(
                    f"unsupported non-local reference '{ref}': only document-"
                    "local references of the form '#/...' are supported"
                ),
                pointer=path_pointer,
                ref=ref,
            )
        )
        return None
    try:
        target = resolver.resolve({"$ref": ref})[0]
    except ReferenceError as exc:
        issues.append(
            _issue(
                "unsupported_reference"
                if isinstance(exc, UnresolvableReferenceError)
                else "circular_reference",
                reason=str(exc),
                pointer=path_pointer,
                ref=getattr(exc, "ref", ref),
            )
        )
        return None
    if not isinstance(target, dict):
        issues.append(
            _issue(
                "invalid_path_item",
                reason=f"path item reference '{ref}' does not target an object",
                pointer=path_pointer,
                ref=ref,
            )
        )
        return None
    merged = dict(target)
    for key, value in raw_item.items():
        if key != "$ref":
            merged[key] = value
    return merged


def _resolve_parameter_entry(
    resolver: RefResolver,
    entry: Any,
    base_pointer: str,
    index: int,
    issues: List[Dict[str, Any]],
) -> Optional[Tuple[Dict[str, Any], str, str]]:
    """Resolve one member of a parameters array.

    Returns ``(parameter, definition_pointer, occurrence)`` where
    *definition_pointer* is the JSON Pointer of the component a ``$ref``
    ultimately reached (``""`` for inline parameters) and *occurrence* is
    the JSON Pointer of the entry in its parameters array.
    """
    occurrence = _join_pointer(base_pointer, index)
    if not isinstance(entry, dict):
        issues.append(
            _issue(
                "invalid_parameter",
                reason="parameter entry must be an object",
                pointer=occurrence,
            )
        )
        return None
    ref = entry.get("$ref")
    if not isinstance(ref, str):
        return entry, "", occurrence
    if not is_local_ref(ref):
        issues.append(
            _issue(
                "unsupported_reference",
                reason=(
                    f"unsupported non-local reference '{ref}': only document-"
                    "local references of the form '#/...' are supported"
                ),
                pointer=occurrence,
                ref=ref,
            )
        )
        return None
    try:
        target, definition = resolver.resolve({"$ref": ref})
    except ReferenceError as exc:
        issues.append(
            _issue(
                "unsupported_reference"
                if isinstance(exc, UnresolvableReferenceError)
                else "circular_reference",
                reason=str(exc),
                pointer=occurrence,
                ref=getattr(exc, "ref", ref),
            )
        )
        return None
    if not isinstance(target, dict):
        issues.append(
            _issue(
                "invalid_parameter",
                reason=f"parameter reference '{ref}' does not target an object",
                pointer=occurrence,
                ref=ref,
            )
        )
        return None
    # In OpenAPI 3.1 a $ref may carry sibling annotations; siblings override
    # referenced members, but $ref itself must not leak into the parameter.
    merged = dict(target)
    for key, value in entry.items():
        if key != "$ref":
            merged[key] = value
    return merged, definition, occurrence


def _dedup_parameters(
    resolved: List[Tuple[Dict[str, Any], str, str]],
    issues: List[Dict[str, Any]],
    *,
    scope: str,
) -> List[Tuple[Dict[str, Any], str, str]]:
    """Drop repeated ``(in, name)`` pairs within one parameter list.

    Each list element is ``(parameter, definition_pointer, occurrence)``.
    Only pairs with a usable ``in`` / ``name`` participate: structurally bad
    entries are reported elsewhere and must not mask one another.
    """
    seen: Dict[Tuple[str, str], str] = {}
    kept: List[Tuple[Dict[str, Any], str, str]] = []
    for parameter, definition, occurrence in resolved:
        location = parameter.get("in")
        name = parameter.get("name")
        if not isinstance(location, str) or not isinstance(name, str):
            kept.append((parameter, definition, occurrence))
            continue
        key = (location, name)
        if key in seen:
            issues.append(
                _issue(
                    "duplicate_parameter",
                    reason=(
                        f"duplicate {scope} parameter "
                        f"({location!r}, {name!r}); a parameter list may "
                        "define each (in, name) pair only once"
                    ),
                    location=location,
                    name=name,
                    pointer=occurrence,
                )
            )
            continue
        seen[key] = occurrence
        kept.append((parameter, definition, occurrence))
    return kept


# --------------------------------------------------------------------------- #
# Primitive / value handling
# --------------------------------------------------------------------------- #

def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _scalar_text(value: Any) -> Optional[str]:
    """Render a JSON scalar the way OpenAPI primitives appear on the wire.

    Returns ``None`` for non-finite floats (NaN/Infinity), which have no
    place in an HTTP message.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return repr(value)
    if isinstance(value, str):
        return value
    return None


def _object_members(value: Dict[str, Any]) -> List[Tuple[str, Any]]:
    return [(str(key), value[key]) for key in sorted(value, key=str)]


def _check_header_text(
    text: str,
    *,
    location: str,
    name: str,
    value_pointer: str,
    issues: List[Dict[str, Any]],
) -> bool:
    """Reject bytes an HTTP header / cookie value cannot carry."""
    bad = next(
        (
            ch
            for ch in text
            if ord(ch) < 0x20 or ord(ch) == 0x7F
        ),
        None,
    )
    if bad is not None:
        issues.append(
            _issue(
                "unsupported_value_type",
                reason=(
                    f"{location} parameter {name!r} renders a control "
                    "character (0x%02X), which HTTP header fields cannot "
                    "carry" % ord(bad)
                ),
                location=location,
                name=name,
                value_pointer=value_pointer,
            )
        )
        return False
    return True


# --------------------------------------------------------------------------- #
# Percent-encoding
# --------------------------------------------------------------------------- #

def _encode_query(text: str, allow_reserved: bool) -> str:
    if allow_reserved:
        # RFC 3986 section 2.2 "reserved" may pass verbatim; quote() always
        # leaves the unreserved set alone anyway.
        return quote(text, safe=":/?#[]@!$&'()*+,;=")
    return quote(text, safe="")


def _encode_path_segment(text: str) -> str:
    return quote(text, safe="!$&'()*+,;=:@-._~")


def _encode_cookie(text: str) -> str:
    # Cookie values get form-style percent encoding (unreserved only).
    return quote(text, safe="")


# --------------------------------------------------------------------------- #
# Parameter validation
# --------------------------------------------------------------------------- #

def _validate_parameter(
    parameter: Dict[str, Any],
    definition: str,
    occurrence: str,
    issues: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Normalize one merged parameter, reporting structural problems."""

    def report(code: str, reason: str) -> None:
        location = parameter.get("in")
        name = parameter.get("name")
        issues.append(
            _issue(
                code,
                reason=reason,
                location=location if isinstance(location, str) else None,
                name=name if isinstance(name, str) else None,
                pointer=occurrence,
                ref=parameter.get("__ref__"),
            )
        )

    location = parameter.get("in")
    name = parameter.get("name")
    if not isinstance(location, str) or location not in PARAM_LOCATIONS:
        report(
            "invalid_parameter",
            f"parameter 'in' must be one of {', '.join(PARAM_LOCATIONS)}",
        )
        return None
    if not isinstance(name, str) or not name:
        report("invalid_parameter", "parameter 'name' must be a non-empty string")
        return None

    if location in ("header", "cookie") and not set(name) <= _TOKEN_CHARS:
        report(
            "invalid_parameter",
            f"{location} parameter name {name!r} is not a valid HTTP token",
        )
        return None

    style = parameter.get("style")
    if style is None:
        style = {
            "path": "simple",
            "query": "form",
            "header": "simple",
            "cookie": "form",
        }[location]
    if not isinstance(style, str) or style not in VALID_STYLES[location]:
        report(
            "invalid_parameter",
            f"style {style!r} is not supported for an {location} parameter "
            f"(valid: {', '.join(sorted(VALID_STYLES[location]))})",
        )
        return None

    explode = parameter.get("explode")
    if explode is None:
        explode = DEFAULT_EXPLODE[(location, style)]
    if not isinstance(explode, bool):
        report("invalid_parameter", "parameter 'explode' must be a boolean")
        return None
    if style in ("spaceDelimited", "pipeDelimited") and explode:
        report(
            "invalid_parameter",
            f"style {style!r} requires explode=false",
        )
        return None
    if style == "deepObject" and not explode:
        report(
            "invalid_parameter",
            "style 'deepObject' requires explode=true",
        )
        return None

    allow_reserved = parameter.get("allowReserved", False)
    if not isinstance(allow_reserved, bool):
        report(
            "invalid_parameter",
            "parameter 'allowReserved' must be a boolean",
        )
        return None
    if allow_reserved and location != "query":
        report(
            "invalid_parameter",
            "allowReserved is only defined for query parameters",
        )
        return None

    required = parameter.get("required", False)
    if not isinstance(required, bool):
        report("invalid_parameter", "parameter 'required' must be a boolean")
        return None
    # Path parameters are always required by specification, regardless of
    # what (or whether) the required flag says.
    if location == "path":
        required = True

    return {
        "in": location,
        "name": name,
        "style": style,
        "explode": explode,
        "allow_reserved": allow_reserved,
        "required": required,
        "occurrence": occurrence,
        "definition": definition,
    }


# --------------------------------------------------------------------------- #
# Serializers
# --------------------------------------------------------------------------- #

def _unsupported(
    *,
    location: str,
    name: str,
    shape: str,
    style: str,
    value_pointer: str,
    issues: List[Dict[str, Any]],
    reason: Optional[str] = None,
) -> None:
    issues.append(
        _issue(
            "unsupported_value_type",
            reason=reason
            or (
                f"{location} parameter {name!r} declares style {style!r}, "
                f"which cannot serialize a {shape} value"
            ),
            location=location,
            name=name,
            value_pointer=value_pointer,
        )
    )


def _primitives(
    values: List[Any],
    encode: Callable[[str], str],
    *,
    location: str,
    name: str,
    base_pointer: str,
    issues: List[Dict[str, Any]],
) -> Optional[List[str]]:
    rendered: List[str] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or isinstance(value, (int, float, str)):
            text = _scalar_text(value)
            if text is None:
                issues.append(
                    _issue(
                        "unsupported_value_type",
                        reason=(
                            f"{location} parameter {name!r} contains a "
                            "non-finite number, which cannot be serialized"
                        ),
                        location=location,
                        name=name,
                        value_pointer=_join_pointer(base_pointer, index),
                    )
                )
                return None
            rendered.append(encode(text))
        else:
            issues.append(
                _issue(
                    "unsupported_value_type",
                    reason=(
                        f"{location} parameter {name!r} contains a non-primitive "
                        "member; only flat arrays of primitives are supported"
                    ),
                    location=location,
                    name=name,
                    value_pointer=_join_pointer(base_pointer, index),
                )
            )
            return None
    return rendered


def render_path_parameter(
    spec: Dict[str, Any], value: Any, value_pointer: str,
    issues: List[Dict[str, Any]],
) -> Optional[str]:
    """Render a path-parameter value into the expression's placeholder."""
    location, name = spec["in"], spec["name"]
    style, explode = spec["style"], spec["explode"]
    encode = _encode_path_segment
    vp = value_pointer

    if _is_scalar(value):
        if value is None:
            _unsupported(location=location, name=name, shape="null",
                         style=style, value_pointer=vp, issues=issues)
            return None
        encoded = encode(_scalar_text(value))
        if style == "simple":
            return encoded
        if style == "label":
            return "." + encoded
        return ";" + name + "=" + encoded

    if isinstance(value, list):
        prim = _primitives(value, encode, location=location, name=name,
                           base_pointer=vp, issues=issues)
        if prim is None:
            return None
        if style == "simple":
            return ",".join(prim)
        if style == "label":
            return "." + ("." if explode else ",").join(prim)
        if explode:
            return "".join(";" + name + "=" + item for item in prim)
        return ";" + name + "=" + ",".join(prim)

    if isinstance(value, dict):
        members = _object_members(value)
        keys = [encode(key) for key, _ in members]
        vals = _primitives(
            [v for _, v in members], encode,
            location=location, name=name, base_pointer=vp, issues=issues,
        )
        if vals is None:
            return None
        pairs = list(zip(keys, vals))
        if style == "simple":
            sep = "=" if explode else ","
            return ",".join(
                (key + sep + val) if explode else (key + "," + val)
                for key, val in pairs
            )
        if style == "label":
            if explode:
                return "." + ",".join(f"{k}={v}" for k, v in pairs)
            return "." + ",".join(
                element for k, v in pairs for element in (k, v)
            )
        # matrix
        if explode:
            # Per the serialization table: ';R=100;L=10' -- a leading
            # semicolon before every key=value pair.
            return "".join(f";{k}={v}" for k, v in pairs)
        return ";" + name + "=" + ",".join(
            element for k, v in pairs for element in (k, v)
        )

    _unsupported(location=location, name=name, shape=type(value).__name__,
                 style=style, value_pointer=vp, issues=issues)
    return None


def render_query_parameter(
    spec: Dict[str, Any], value: Any, value_pointer: str,
    issues: List[Dict[str, Any]],
) -> Optional[List[str]]:
    """Render one query parameter into ``key=value`` pair strings."""
    location, name = spec["in"], spec["name"]
    style, explode = spec["style"], spec["explode"]
    allow_reserved = spec["allow_reserved"]

    def encode(text: str) -> str:
        return _encode_query(text, allow_reserved)

    enc_name = quote(name, safe="")

    if style == "deepObject":
        if not isinstance(value, dict):
            _unsupported(
                location=location, name=name,
                shape=type(value).__name__ if value is not None else "null",
                style=style, value_pointer=value_pointer, issues=issues,
                reason=(
                    "query parameter " + repr(name) + " declares style "
                    "'deepObject', which requires an object value"
                ),
            )
            return None
        pairs: List[str] = []
        for key, member in _object_members(value):
            member_pointer = _join_pointer(value_pointer, key)
            if not _is_scalar(member) or member is None or isinstance(
                member, (dict, list)
            ):
                _unsupported(
                    location=location, name=name,
                    shape="nested "
                    + ("object" if isinstance(member, dict) else "array"),
                    style=style, value_pointer=member_pointer, issues=issues,
                    reason=(
                        "deepObject query parameter " + repr(name)
                        + " supports only primitive object members"
                    ),
                )
                return None
            pairs.append(
                f"{enc_name}%5B{quote(key, safe='')}%5D="
                + encode(_scalar_text(member))
            )
        return pairs

    # Structural separators emitted per the serialization table. A space is
    # illegal in a request target and appears percent-encoded (the spec's
    # spaceDelimited examples show t%20a%20g); the pipe stays raw.
    arr_delim = {
        "form": ",",
        "spaceDelimited": "%20",
        "pipeDelimited": "|",
    }[style]

    if _is_scalar(value):
        if value is None:
            _unsupported(location=location, name=name, shape="null",
                         style=style, value_pointer=value_pointer,
                         issues=issues)
            return None
        if style != "form":
            _unsupported(
                location=location, name=name, shape="primitive",
                style=style, value_pointer=value_pointer, issues=issues,
                reason=(
                    f"query parameter {name!r} declares style {style!r}, "
                    "which is only defined for arrays"
                ),
            )
            return None
        return [enc_name + "=" + encode(_scalar_text(value))]

    if isinstance(value, list):
        if style != "form" and explode:
            # Already rejected in validation, but keep the guard explicit.
            _unsupported(location=location, name=name, shape="array",
                         style=style, value_pointer=value_pointer,
                         issues=issues)
            return None
        prim = _primitives(value, encode, location=location, name=name,
                           base_pointer=value_pointer, issues=issues)
        if prim is None:
            return None
        if style == "form" and explode:
            return [enc_name + "=" + item for item in prim]
        if style == "form":
            return [enc_name + "=" + ",".join(prim)]
        # spaceDelimited / pipeDelimited: name once, raw separator.
        return [enc_name + "=" + arr_delim.join(prim)]

    if isinstance(value, dict):
        if style != "form":
            _unsupported(
                location=location, name=name,
                shape="object", style=style,
                value_pointer=value_pointer, issues=issues,
                reason=(
                    f"query parameter {name!r} declares style {style!r}, "
                    "which is only defined for arrays"
                ),
            )
            return None
        members = _object_members(value)
        keys = [quote(key, safe="") for key, _ in members]
        vals = _primitives(
            [v for _, v in members], encode,
            location=location, name=name,
            base_pointer=value_pointer, issues=issues,
        )
        if vals is None:
            return None
        pairs_obj = list(zip(keys, vals))
        if explode:
            return [f"{k}={v}" for k, v in pairs_obj]
        joined = ",".join(
            element for k, v in pairs_obj for element in (k, v)
        )
        return [enc_name + "=" + joined]

    _unsupported(location=location, name=name, shape=type(value).__name__,
                 style=style, value_pointer=value_pointer, issues=issues)
    return None


def render_header_parameter(
    spec: Dict[str, Any], value: Any, value_pointer: str,
    issues: List[Dict[str, Any]],
) -> Optional[str]:
    """Render a header parameter's value (simple style, no percent encoding)."""
    location, name = spec["in"], spec["name"]

    def encode(text: str) -> str:
        return text

    if _is_scalar(value):
        if value is None:
            _unsupported(location=location, name=name, shape="null",
                         style="simple", value_pointer=value_pointer,
                         issues=issues)
            return None
        text = _scalar_text(value)
        if not _check_header_text(text, location=location, name=name,
                                  value_pointer=value_pointer, issues=issues):
            return None
        return text

    if isinstance(value, list):
        prim = _primitives(value, encode, location=location, name=name,
                           base_pointer=value_pointer, issues=issues)
        if prim is None:
            return None
        rendered = ",".join(prim)
        if not _check_header_text(rendered, location=location, name=name,
                                  value_pointer=value_pointer, issues=issues):
            return None
        return rendered

    if isinstance(value, dict):
        members = _object_members(value)
        keys = [key for key, _ in members]
        vals = _primitives(
            [v for _, v in members], encode,
            location=location, name=name,
            base_pointer=value_pointer, issues=issues,
        )
        if vals is None:
            return None
        if spec["explode"]:
            rendered = ",".join(f"{k}={v}" for k, v in zip(keys, vals))
        else:
            rendered = ",".join(
                element for k, v in zip(keys, vals)
                for element in (k, v)
            )
        if not _check_header_text(rendered, location=location, name=name,
                                  value_pointer=value_pointer, issues=issues):
            return None
        return rendered

    _unsupported(location=location, name=name, shape=type(value).__name__,
                 style="simple", value_pointer=value_pointer, issues=issues)
    return None


def render_cookie_parameter(
    spec: Dict[str, Any], value: Any, value_pointer: str,
    issues: List[Dict[str, Any]],
) -> Optional[str]:
    """Render a cookie parameter (form style, explode defaults to false)."""
    location, name = spec["in"], spec["name"]
    enc_name = quote(name, safe="")

    if _is_scalar(value):
        if value is None:
            _unsupported(location=location, name=name, shape="null",
                         style="form", value_pointer=value_pointer,
                         issues=issues)
            return None
        text = _scalar_text(value)
        if not _check_header_text(text, location=location, name=name,
                                  value_pointer=value_pointer, issues=issues):
            return None
        return f"{enc_name}={_encode_cookie(text)}"

    if isinstance(value, list):
        prim = _primitives(value, _encode_cookie, location=location,
                           name=name, base_pointer=value_pointer,
                           issues=issues)
        if prim is None:
            return None
        if spec["explode"]:
            body = "&".join(f"{enc_name}={item}" for item in prim)
        else:
            body = f"{enc_name}=" + ",".join(prim)
        if not _check_header_text(body, location=location, name=name,
                                  value_pointer=value_pointer, issues=issues):
            return None
        return body

    if isinstance(value, dict):
        members = _object_members(value)
        keys = [quote(key, safe="") for key, _ in members]
        vals = _primitives(
            [v for _, v in members], _encode_cookie,
            location=location, name=name,
            base_pointer=value_pointer, issues=issues,
        )
        if vals is None:
            return None
        pairs = list(zip(keys, vals))
        if spec["explode"]:
            # form explode=true uses the object's member keys (no repeated
            # parameter name), exactly like a query form object.
            body = "&".join(f"{k}={v}" for k, v in pairs)
        else:
            body = enc_name + "=" + ",".join(
                element for k, v in pairs for element in (k, v)
            )
        if not _check_header_text(body, location=location, name=name,
                                  value_pointer=value_pointer, issues=issues):
            return None
        return body

    _unsupported(location=location, name=name, shape=type(value).__name__,
                 style="form", value_pointer=value_pointer, issues=issues)
    return None


# --------------------------------------------------------------------------- #
# End-to-end materialization
# --------------------------------------------------------------------------- #

def materialize_request(
    document: Any,
    raw_selector: Any,
    values: Any = None,
) -> Dict[str, str]:
    """Build ``request_target``/``headers``/``cookie`` for one operation.

    :raises InvalidMaterializeRequestError: bad document shape, malformed
        selector / unknown operation, or malformed ``values`` grouping (400).
    :raises MaterializationError: the contract + values cannot be rendered
        (missing required values, unknown supplied values, duplicate
        parameters, dangling/circular references, unsupported values) (422).
    """
    shape_messages = validate_openapi_shape(document)
    if shape_messages:
        raise InvalidMaterializeRequestError(shape_messages)

    selected = parse_selector(raw_selector)
    if selected is None:
        raise InvalidMaterializeRequestError(
            [
                f"malformed selector {raw_selector!r}: expected "
                "'METHOD /path' (e.g. 'GET /pets/{id}')"
            ]
        )
    method, path = selected

    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise InvalidMaterializeRequestError(
            ["values must be an object keyed by parameter location"]
        )
    bad_groups = [
        key for key in values if key not in PARAM_LOCATIONS
    ]
    if bad_groups:
        raise InvalidMaterializeRequestError(
            [
                f"unknown value group {key!r}: groups must be one of "
                f"{', '.join(PARAM_LOCATIONS)}"
                for key in sorted(bad_groups, key=str)
            ]
        )
    for group in PARAM_LOCATIONS:
        group_value = values.get(group, {})
        if not isinstance(group_value, dict):
            raise InvalidMaterializeRequestError(
                [f"values.{group} must be an object of parameter names"]
            )

    paths = document["paths"]
    if not isinstance(paths, dict) or path not in paths:
        raise InvalidMaterializeRequestError(
            [f"unknown operation: {method.upper()} {path}: path is not declared"]
        )

    issues: List[Dict[str, Any]] = []
    resolver = RefResolver(document, label="document")
    path_pointer = _pointer("paths", path)

    item = _resolve_path_item(resolver, paths[path], path_pointer, issues)
    if item is None:
        raise MaterializationError(_sort_issues(issues))

    actual_key = next(
        (k for k in item if k.lower() == method and k.lower() in HTTP_METHODS),
        None,
    )
    if actual_key is None:
        raise InvalidMaterializeRequestError(
            [f"unknown operation: {method.upper()} {path}: operation is not declared"]
        )
    operation = item[actual_key]
    if not isinstance(operation, dict):
        raise InvalidMaterializeRequestError(
            [f"unknown operation: {method.upper()} {path}: operation is not an object"]
        )
    operation_pointer = _join_pointer(path_pointer, actual_key.lower())

    # ----- resolve parameter lists -----------------------------------------
    raw_path_params = item.get("parameters", [])
    raw_op_params = operation.get("parameters", [])
    if not isinstance(raw_path_params, list):
        issues.append(
            _issue(
                "invalid_parameter",
                reason="path-level 'parameters' must be an array",
                pointer=_join_pointer(path_pointer, "parameters"),
            )
        )
        raw_path_params = []
    if not isinstance(raw_op_params, list):
        issues.append(
            _issue(
                "invalid_parameter",
                reason="operation 'parameters' must be an array",
                pointer=_join_pointer(operation_pointer, "parameters"),
            )
        )
        raw_op_params = []

    path_resolved = [
        result
        for index, entry in enumerate(raw_path_params)
        if (
            result := _resolve_parameter_entry(
                resolver,
                entry,
                _join_pointer(path_pointer, "parameters"),
                index,
                issues,
            )
        )
        is not None
    ]
    op_resolved = [
        result
        for index, entry in enumerate(raw_op_params)
        if (
            result := _resolve_parameter_entry(
                resolver,
                entry,
                _join_pointer(operation_pointer, "parameters"),
                index,
                issues,
            )
        )
        is not None
    ]

    # Tag parameters that arrived via $ref so validation issues can echo the
    # reference, then duplicate-check each list.
    def tag_refs(
        rows: List[Tuple[Dict[str, Any], str, str]],
    ) -> List[Tuple[Dict[str, Any], str, str]]:
        tagged = []
        for parameter, definition, occurrence in rows:
            if definition:
                parameter = dict(parameter)
                parameter["__ref__"] = "#" + definition
            tagged.append((parameter, definition, occurrence))
        return tagged

    path_resolved = tag_refs(path_resolved)
    op_resolved = tag_refs(op_resolved)

    path_rows = _dedup_parameters(path_resolved, issues, scope="path-level")
    op_rows = _dedup_parameters(op_resolved, issues, scope="operation-level")

    # Merge: an operation-level parameter with key (in, name) overrides the
    # path-level one instead of colliding with it.
    op_keys = {
        (p.get("in"), p.get("name"))
        for p, _, _ in op_rows
        if isinstance(p.get("in"), str) and isinstance(p.get("name"), str)
    }
    merged_rows: List[Tuple[Dict[str, Any], str, str]] = [
        row
        for row in path_rows
        if (row[0].get("in"), row[0].get("name")) not in op_keys
    ]
    merged_rows.extend(op_rows)

    specs: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for parameter, definition, occurrence in merged_rows:
        normalized = _validate_parameter(
            parameter, definition, occurrence, issues
        )
        if normalized is not None:
            specs[(normalized["in"], normalized["name"])] = normalized

    if issues:
        raise MaterializationError(_sort_issues(issues))

    # ----- supplied values --------------------------------------------------
    def supplied(location: str) -> Dict[str, Any]:
        return values.get(location, {})

    # Unknown values: a name supplied for which no parameter exists.
    for location in PARAM_LOCATIONS:
        known = {n for (loc, n) in specs if loc == location}
        for name in sorted(supplied(location), key=str):
            if name not in known:
                issues.append(
                    _issue(
                        "unknown_value",
                        reason=(
                            f"{location} parameter {name!r} is not declared "
                            "by the selected operation"
                        ),
                        location=location,
                        name=name,
                        value_pointer=_pointer("values", location, name),
                    )
                )

    # Null / JSON-type sanity plus required handling happens while rendering.
    def get_value(location: str, name: str) -> Tuple[bool, Any, str]:
        group = supplied(location)
        present = name in group
        return present, group.get(name) if present else None, _pointer(
            "values", location, name
        )

    # Required checks first so a missing required parameter is reported even
    # if other rendering fails.
    for (location, name), spec in sorted(specs.items()):
        present, value, vp = get_value(location, name)
        if not present and spec["required"]:
            issues.append(
                _issue(
                    "missing_required_value",
                    reason=(
                        f"{location} parameter {name!r} is required but no "
                        "value was supplied"
                    ),
                    location=location,
                    name=name,
                    pointer=spec["occurrence"],
                )
            )
        elif present and value is None:
            issues.append(
                _issue(
                    "unsupported_value_type",
                    reason=(
                        f"{location} parameter {name!r} has JSON null value; "
                        "a null parameter value cannot be serialized"
                    ),
                    location=location,
                    name=name,
                    value_pointer=vp,
                )
            )

    if issues:
        raise MaterializationError(_sort_issues(issues))

    # ----- render -----------------------------------------------------------
    path_substitutions: List[Tuple[str, str]] = []
    query_pairs: List[Tuple[str, List[str]]] = []
    headers: List[Tuple[str, str]] = []
    cookies: List[Tuple[str, str]] = []

    for (location, name), spec in sorted(specs.items()):
        present, value, vp = get_value(location, name)
        if not present:
            continue
        if location == "path":
            rendered = render_path_parameter(spec, value, vp, issues)
            if rendered is not None:
                path_substitutions.append((name, rendered))
        elif location == "query":
            rendered = render_query_parameter(spec, value, vp, issues)
            if rendered is not None:
                query_pairs.append((name, rendered))
        elif location == "header":
            rendered = render_header_parameter(spec, value, vp, issues)
            if rendered is not None:
                headers.append((name, rendered))
        else:  # cookie
            rendered = render_cookie_parameter(spec, value, vp, issues)
            if rendered is not None:
                cookies.append((name, rendered))

    if issues:
        raise MaterializationError(_sort_issues(issues))

    # ----- assemble ---------------------------------------------------------
    missing_placeholders: List[str] = []

    def substitute(match: "re.Match[str]") -> str:
        placeholder = match.group(1)
        # Ignore any RFC 6570 operators; plain OpenAPI templates never use them.
        for name, rendered in path_substitutions:
            if placeholder == name:
                return rendered
        missing_placeholders.append(placeholder)
        return match.group(0)

    target_path = _PATH_TEMPLATE.sub(substitute, path)
    if missing_placeholders:
        for placeholder in sorted(set(missing_placeholders)):
            issues.append(
                _issue(
                    "unresolved_path_template",
                    reason=(
                        f"path template placeholder {placeholder!r} has no "
                        "matching path parameter"
                    ),
                    location="path",
                    name=placeholder,
                    pointer=path_pointer,
                )
            )
        raise MaterializationError(_sort_issues(issues))

    # Duplicated query keys are kept (arrays repeat naturally); ordering is
    # by parameter name and within a parameter by the rendered pair order.
    query = "&".join(
        pair
        for _name, pairs in sorted(query_pairs, key=lambda item: item[0])
        for pair in pairs
    )
    request_target = target_path + ("?" + query if query else "")

    headers_out = {
        name: value for name, value in sorted(headers, key=lambda item: item[0])
    }
    cookie = "; ".join(
        value for _name, value in sorted(cookies, key=lambda item: item[0])
    )

    return {
        "request_target": request_target,
        "headers": headers_out,
        "cookie": cookie,
    }
