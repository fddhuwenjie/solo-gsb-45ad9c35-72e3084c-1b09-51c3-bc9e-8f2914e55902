"""Regression tests for ``POST /requests/materialize``."""

from __future__ import annotations

import copy

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def make_doc(paths=None, components=None, **extra) -> dict:
    doc = {
        "openapi": "3.0.3",
        "info": {"title": "Pet API", "version": "1.0"},
        "paths": paths or {},
    }
    if components:
        doc["components"] = components
    doc.update(extra)
    return doc


def op(responses=None, **extra) -> dict:
    operation = {"responses": responses or {"200": {"description": "ok"}}}
    operation.update(extra)
    return operation


def materialize(doc, selector, values=None):
    body: dict = {"document": doc, "selector": selector}
    if values is not None:
        body["values"] = values
    return client.post("/requests/materialize", json=body)


def param(name, location, **extra):
    parameter = {"name": name, "in": location}
    parameter.update(extra)
    return parameter


# --------------------------------------------------------------------------- #
# Happy path basics
# --------------------------------------------------------------------------- #

def test_minimal_operation_without_parameters():
    doc = make_doc({"/ping": {"get": op()}})
    resp = materialize(doc, "GET /ping")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "request_target": "/ping",
        "headers": {},
        "cookie": "",
    }


def test_method_matching_is_case_insensitive():
    doc = make_doc({"/p": {"GeT": op()}})
    resp = materialize(doc, "get /p")
    assert resp.status_code == 200
    assert resp.json()["request_target"] == "/p"


def test_path_simple_is_the_default_for_path_parameters():
    doc = make_doc({
        "/p/{id}": {"get": op(parameters=[
            param("id", "path", required=True),
        ])},
    })
    resp = materialize(doc, "GET /p/{id}", {"path": {"id": "a/b c"}})
    assert resp.status_code == 200, resp.text
    assert resp.json()["request_target"] == "/p/a%2Fb%20c"


def test_booleans_integers_and_floats_render_as_their_json_text():
    doc = make_doc({
        "/p/{a}/{b}/{c}": {"get": op(parameters=[
            param("a", "path", required=True),
            param("b", "path", required=True),
            param("c", "path", required=True),
        ])},
    })
    resp = materialize(
        doc, "GET /p/{a}/{b}/{c}",
        {"path": {"a": True, "b": -3, "c": 1.5}},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["request_target"] == "/p/true/-3/1.5"


def test_response_shape_always_has_three_fields():
    doc = make_doc({"/p": {"get": op()}})
    body = materialize(doc, "GET /p").json()
    assert set(body) == {"request_target", "headers", "cookie"}


# --------------------------------------------------------------------------- #
# Path styles
# --------------------------------------------------------------------------- #

def test_path_simple_array_explode_false_is_comma():
    # simple defaults to explode=false.
    doc = make_doc({"/p/{ids}": {"get": op(parameters=[
        param("ids", "path", required=True,
              schema={"type": "array", "items": {"type": "string"}}),
    ])}})
    resp = materialize(doc, "GET /p/{ids}", {"path": {"ids": ["a", "b", "c"]}})
    assert resp.status_code == 200, resp.text
    assert resp.json()["request_target"] == "/p/a,b,c"


def test_path_simple_object_explode_true():
    doc = make_doc({"/p/{coords}": {"get": op(parameters=[
        param("coords", "path", required=True, style="simple", explode=True,
              schema={"type": "object"}),
    ])}})
    resp = materialize(
        doc, "GET /p/{coords}",
        {"path": {"coords": {"x": "1", "y": "2"}}},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["request_target"] == "/p/x=1,y=2"


def test_path_simple_object_explode_false_flattens_keys_and_values():
    doc = make_doc({"/p/{coords}": {"get": op(parameters=[
        param("coords", "path", required=True, style="simple", explode=False,
              schema={"type": "object"}),
    ])}})
    resp = materialize(
        doc, "GET /p/{coords}",
        {"path": {"coords": {"y": "2", "x": "1"}}},
    )
    # Keys sorted -> output order independent of input key order.
    assert resp.json()["request_target"] == "/p/x,1,y,2"


def test_path_label_primitive():
    doc = make_doc({"/p/{id}": {"get": op(parameters=[
        param("id", "path", required=True, style="label"),
    ])}})
    resp = materialize(doc, "GET /p/{id}", {"path": {"id": "v"}})
    assert resp.json()["request_target"] == "/p/.v"


def test_path_label_array_explode_false_uses_one_dot():
    doc = make_doc({"/p/{ids}": {"get": op(parameters=[
        param("ids", "path", required=True, style="label"),
    ])}})
    resp = materialize(doc, "GET /p/{ids}", {"path": {"ids": ["a", "b"]}})
    assert resp.json()["request_target"] == "/p/.a,b"


def test_path_label_array_explode_true_dots_every_value():
    doc = make_doc({"/p/{ids}": {"get": op(parameters=[
        param("ids", "path", required=True, style="label", explode=True),
    ])}})
    resp = materialize(doc, "GET /p/{ids}", {"path": {"ids": ["a", "b"]}})
    assert resp.json()["request_target"] == "/p/.a.b"


def test_path_label_object_explode_true():
    doc = make_doc({"/p/{o}": {"get": op(parameters=[
        param("o", "path", required=True, style="label", explode=True),
    ])}})
    resp = materialize(doc, "GET /p/{o}",
                       {"path": {"o": {"b": "2", "a": "1"}}})
    assert resp.json()["request_target"] == "/p/.a=1,b=2"


def test_path_label_object_explode_false():
    doc = make_doc({"/p/{o}": {"get": op(parameters=[
        param("o", "path", required=True, style="label", explode=False),
    ])}})
    resp = materialize(doc, "GET /p/{o}",
                       {"path": {"o": {"b": "2", "a": "1"}}})
    assert resp.json()["request_target"] == "/p/.a,1,b,2"


def test_path_matrix_primitive():
    doc = make_doc({"/p/{id}": {"get": op(parameters=[
        param("id", "path", required=True, style="matrix"),
    ])}})
    resp = materialize(doc, "GET /p/{id}", {"path": {"id": "v"}})
    # matrix defaults to explode=true but primitive rendering is identical.
    assert resp.json()["request_target"] == "/p/;id=v"


def test_path_matrix_array_explode_true_repeats_semicolon_name():
    doc = make_doc({"/p/{ids}": {"get": op(parameters=[
        param("ids", "path", required=True, style="matrix", explode=True),
    ])}})
    resp = materialize(doc, "GET /p/{ids}",
                       {"path": {"ids": ["a", "b"]}})
    assert resp.json()["request_target"] == "/p/;ids=a;ids=b"


def test_path_matrix_array_explode_false_single_assignment():
    doc = make_doc({"/p/{ids}": {"get": op(parameters=[
        param("ids", "path", required=True, style="matrix", explode=False),
    ])}})
    resp = materialize(doc, "GET /p/{ids}",
                       {"path": {"ids": ["a", "b"]}})
    assert resp.json()["request_target"] == "/p/;ids=a,b"


def test_path_matrix_object_explode_true():
    doc = make_doc({"/p/{o}": {"get": op(parameters=[
        param("o", "path", required=True, style="matrix", explode=True),
    ])}})
    resp = materialize(doc, "GET /p/{o}",
                       {"path": {"o": {"b": "2", "a": "1"}}})
    assert resp.json()["request_target"] == "/p/;a=1;b=2"


def test_path_matrix_object_explode_false():
    doc = make_doc({"/p/{o}": {"get": op(parameters=[
        param("o", "path", required=True, style="matrix", explode=False),
    ])}})
    resp = materialize(doc, "GET /p/{o}",
                       {"path": {"o": {"b": "2", "a": "1"}}})
    assert resp.json()["request_target"] == "/p/;o=a,1,b,2"


def test_path_percent_encoding_covers_slashes_spaces_and_unicode():
    doc = make_doc({"/p/{id}": {"get": op(parameters=[
        param("id", "path", required=True),
    ])}})
    resp = materialize(doc, "GET /p/{id}", {"path": {"id": "a/b c%€"}})
    assert resp.status_code == 200
    target = resp.json()["request_target"]
    assert target == "/p/a%2Fb%20c%25%E2%82%AC"


def test_path_style_allowed_sub_delims_pass_but_brackets_encode():
    doc = make_doc({"/p/{id}": {"get": op(parameters=[
        param("id", "path", required=True),
    ])}})
    resp = materialize(doc, "GET /p/{id}", {"path": {"id": "a!$'x[]"}})
    assert resp.json()["request_target"] == "/p/a!$'x%5B%5D"


def test_path_template_may_repeat_one_parameter():
    doc = make_doc({"/p/{x}/{x}": {"get": op(parameters=[
        param("x", "path", required=True),
    ])}})
    resp = materialize(doc, "GET /p/{x}/{x}", {"path": {"x": "z"}})
    assert resp.status_code == 200
    assert resp.json()["request_target"] == "/p/z/z"


# --------------------------------------------------------------------------- #
# Query styles
# --------------------------------------------------------------------------- #

def test_query_form_primitive_is_default():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("q", "query"),
    ])}})
    resp = materialize(doc, "GET /p", {"query": {"q": "a b"}})
    assert resp.json()["request_target"] == "/p?q=a%20b"


def test_query_form_array_explode_true_repeats_keys():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("t", "query", schema={"type": "array"}),
    ])}})
    resp = materialize(doc, "GET /p",
                       {"query": {"t": ["a b", "c,d", "x/y"]}})
    assert resp.json()["request_target"] == (
        "/p?t=a%20b&t=c%2Cd&t=x%2Fy"
    )


def test_query_form_array_explode_false_single_key_csv():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("t", "query", explode=False, schema={"type": "array"}),
    ])}})
    resp = materialize(doc, "GET /p", {"query": {"t": ["a", "b", "c"]}})
    assert resp.json()["request_target"] == "/p?t=a,b,c"


def test_query_form_object_explode_true_uses_member_keys():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("f", "query", schema={"type": "object"}),
    ])}})
    resp = materialize(doc, "GET /p",
                       {"query": {"f": {"z": "9", "a": "1"}}})
    assert resp.json()["request_target"] == "/p?a=1&z=9"


def test_query_form_object_explode_false_flattens_under_parameter_name():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("f", "query", explode=False, schema={"type": "object"}),
    ])}})
    resp = materialize(doc, "GET /p",
                       {"query": {"f": {"z": "9", "a": "1"}}})
    assert resp.json()["request_target"] == "/p?f=a,1,z,9"


def test_query_space_delimited_array():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("t", "query", style="spaceDelimited",
              schema={"type": "array"}),
    ])}})
    resp = materialize(doc, "GET /p", {"query": {"t": ["a", "b c"]}})
    assert resp.json()["request_target"] == "/p?t=a%20b%20c"


def test_query_pipe_delimited_array():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("t", "query", style="pipeDelimited",
              schema={"type": "array"}),
    ])}})
    resp = materialize(doc, "GET /p", {"query": {"t": ["a", "b|c"]}})
    # The delimiter pipe stays raw; a pipe inside a value is encoded.
    assert resp.json()["request_target"] == "/p?t=a|b%7Cc"


def test_query_deep_object_object():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("f", "query", style="deepObject",
              schema={"type": "object"}),
    ])}})
    resp = materialize(
        doc, "GET /p",
        {"query": {"f": {"role": "a d[m]", "n": 2}}},
    )
    assert resp.json()["request_target"] == (
        "/p?f%5Bn%5D=2&f%5Brole%5D=a%20d%5Bm%5D"
    )


def test_query_parameters_sorted_by_name_regardless_of_input_order():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("z", "query"), param("a", "query"), param("m", "query"),
    ])}})
    resp = materialize(doc, "GET /p",
                       {"query": {"m": "1", "a": "2", "z": "3"}})
    assert resp.json()["request_target"] == "/p?a=2&m=1&z=3"


def test_repeated_query_keys_from_one_array_stay_contiguous():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("t", "query", schema={"type": "array"}),
        param("a", "query"),
    ])}})
    resp = materialize(doc, "GET /p",
                       {"query": {"t": ["1", "2", "3"], "a": "x"}})
    assert resp.json()["request_target"] == "/p?a=x&t=1&t=2&t=3"


def test_allow_reserved_lets_reserved_characters_pass_raw():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("q", "query", allowReserved=True),
    ])}})
    resp = materialize(doc, "GET /p",
                       {"query": {"q": "a/b?c=d&e#f[]:g"}})
    assert resp.status_code == 200, resp.text
    assert resp.json()["request_target"] == "/p?q=a/b?c=d&e#f[]:g"


def test_without_allow_reserved_reserved_characters_are_encoded():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("q", "query"),
    ])}})
    resp = materialize(doc, "GET /p", {"query": {"q": "a/b?c"}})
    assert resp.json()["request_target"] == "/p?q=a%2Fb%3Fc"


def test_empty_string_primitive_renders_as_empty_value():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("q", "query"),
    ])}})
    resp = materialize(doc, "GET /p", {"query": {"q": ""}})
    assert resp.json()["request_target"] == "/p?q="


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

def test_header_simple_primitive_is_not_percent_encoded():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("X-Trace", "header"),
    ])}})
    resp = materialize(doc, "GET /p",
                       {"header": {"X-Trace": "a/b,c d"}})
    assert resp.json()["request_target"] == "/p"
    assert resp.json()["headers"] == {"X-Trace": "a/b,c d"}


def test_header_simple_array_default_explode_false_is_csv():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("X-T", "header", schema={"type": "array"}),
    ])}})
    resp = materialize(doc, "GET /p",
                       {"header": {"X-T": ["a", "b"]}})
    assert resp.json()["headers"] == {"X-T": "a,b"}


def test_header_simple_object_explode_true_and_false():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("X-A", "header", style="simple", explode=True),
        param("X-B", "header", style="simple", explode=False),
    ])}})
    resp = materialize(
        doc, "GET /p",
        {"header": {
            "X-A": {"b": "2", "a": "1"},
            "X-B": {"b": "2", "a": "1"},
        }},
    )
    assert resp.json()["headers"] == {"X-A": "a=1,b=2", "X-B": "a,1,b,2"}


def test_headers_are_sorted_by_name():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("Z", "header"), param("A", "header"), param("M", "header"),
    ])}})
    resp = materialize(doc, "GET /p",
                       {"header": {"Z": "1", "M": "2", "A": "3"}})
    assert list(resp.json()["headers"]) == ["A", "M", "Z"]


def test_cookie_parameter_never_leaks_into_headers():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("X-A", "header"),
        param("sess", "cookie"),
    ])}})
    resp = materialize(
        doc, "GET /p",
        {"header": {"X-A": "1"}, "cookie": {"sess": "abc"}},
    )
    assert set(resp.json()["headers"]) == {"X-A"}
    assert resp.json()["cookie"] == "sess=abc"


# --------------------------------------------------------------------------- #
# Cookie
# --------------------------------------------------------------------------- #

def test_cookie_form_primitive_percent_encoded():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("sess", "cookie"),
    ])}})
    resp = materialize(doc, "GET /p",
                       {"cookie": {"sess": "a b/c=d"}})
    assert resp.json()["cookie"] == "sess=a%20b%2Fc%3Dd"


def test_cookie_form_array_default_explode_false_is_csv():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("ids", "cookie", schema={"type": "array"}),
    ])}})
    resp = materialize(doc, "GET /p",
                       {"cookie": {"ids": ["a", "b"]}})
    assert resp.json()["cookie"] == "ids=a,b"


def test_cookie_form_array_explode_true_uses_ampersand():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("ids", "cookie", explode=True, schema={"type": "array"}),
    ])}})
    resp = materialize(doc, "GET /p",
                       {"cookie": {"ids": ["a", "b"]}})
    assert resp.json()["cookie"] == "ids=a&ids=b"


def test_cookie_form_object_is_explode_false_csv_flatten():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("g", "cookie"),
    ])}})
    resp = materialize(
        doc, "GET /p",
        {"cookie": {"g": {"b": "2", "a": "1"}}},
    )
    assert resp.json()["cookie"] == "g=a,1,b,2"


def test_cookie_form_object_explicit_explode_true_uses_member_keys():
    # explode=true is not part of the cookie column of the serialization
    # table, but the form style's explode rule (member keys + '&') is what
    # such an explicit declaration produces.
    doc = make_doc({"/p": {"get": op(parameters=[
        param("f", "cookie", explode=True),
    ])}})
    resp = materialize(
        doc, "GET /p",
        {"cookie": {"f": {"b": "2", "a": "1"}}},
    )
    assert resp.json()["cookie"] == "a=1&b=2"


def test_multiple_cookies_joined_with_semicolon_and_sorted():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("z", "cookie"), param("a", "cookie"),
    ])}})
    resp = materialize(doc, "GET /p",
                       {"cookie": {"z": "1", "a": "2"}})
    assert resp.json()["cookie"] == "a=2; z=1"


# --------------------------------------------------------------------------- #
# Path-level / operation-level merging
# --------------------------------------------------------------------------- #

def test_path_level_and_operation_parameters_are_merged():
    doc = make_doc({
        "/p/{id}": {
            "parameters": [
                param("id", "path", required=True),
                param("X-Shared", "header"),
            ],
            "get": op(parameters=[
                param("q", "query"),
            ]),
        },
    })
    resp = materialize(
        doc, "GET /p/{id}",
        {"path": {"id": "7"}, "query": {"q": "x"},
         "header": {"X-Shared": "h"}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["request_target"] == "/p/7?q=x"
    assert body["headers"] == {"X-Shared": "h"}


def test_operation_parameter_overrides_path_parameter_by_in_name():
    # The path-level header is CSV-explode; the operation-level header with
    # the same name overrides it with an object/explode declaration.
    doc = make_doc({
        "/p": {
            "parameters": [param("X-F", "header")],
            "get": op(parameters=[
                param("X-F", "header", style="simple", explode=True),
            ]),
        },
    })
    resp = materialize(
        doc, "GET /p",
        {"header": {"X-F": {"b": "2", "a": "1"}}},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["headers"] == {"X-F": "a=1,b=2"}


def test_override_also_applies_across_references():
    doc = make_doc(
        {
            "/p": {
                "parameters": [{"$ref": "#/components/parameters/Tenant"}],
                "get": op(parameters=[
                    param("X-Tenant", "header",
                          description="operation override"),
                ]),
            },
        },
        components={"parameters": {
            "Tenant": param("X-Tenant", "header", required=True),
        }},
    )
    resp = materialize(doc, "GET /p",
                       {"header": {"X-Tenant": "t"}})
    assert resp.status_code == 200, resp.text
    assert resp.json()["headers"] == {"X-Tenant": "t"}


# --------------------------------------------------------------------------- #
# Local $ref resolution
# --------------------------------------------------------------------------- #

def test_parameter_references_resolve_from_components():
    doc = make_doc(
        {
            "/p/{id}": {"get": op(parameters=[
                {"$ref": "#/components/parameters/Id"},
                {"$ref": "#/components/parameters/Q"},
            ])},
        },
        components={"parameters": {
            "Id": param("id", "path", required=True),
            "Q": param("q", "query"),
        }},
    )
    resp = materialize(doc, "GET /p/{id}",
                       {"path": {"id": "9"}, "query": {"q": "x"}})
    assert resp.status_code == 200, resp.text
    assert resp.json()["request_target"] == "/p/9?q=x"


def test_ref_chains_are_followed():
    doc = make_doc(
        {
            "/p/{id}": {"get": op(parameters=[
                {"$ref": "#/components/parameters/IdAlias"},
            ])},
        },
        components={"parameters": {
            "IdAlias": {"$ref": "#/components/parameters/Id"},
            "Id": param("id", "path", required=True),
        }},
    )
    resp = materialize(doc, "GET /p/{id}", {"path": {"id": "9"}})
    assert resp.status_code == 200, resp.text
    assert resp.json()["request_target"] == "/p/9"


def test_dangling_reference_is_422_with_occurrence_pointer():
    doc = make_doc({
        "/p": {"get": op(parameters=[
            {"$ref": "#/components/parameters/Missing"},
        ])},
    })
    resp = materialize(doc, "GET /p")
    assert resp.status_code == 422
    issue = resp.json()["issues"][0]
    assert issue["code"] == "unsupported_reference"
    assert issue["ref"] == "#/components/parameters/Missing"
    assert issue["pointer"] == "/paths/~1p/get/parameters/0"


def test_non_local_reference_is_422():
    doc = make_doc({
        "/p": {"get": op(parameters=[
            {"$ref": "https://example.com/params.json#/Id"},
        ])},
    })
    resp = materialize(doc, "GET /p")
    assert resp.status_code == 422
    issue = resp.json()["issues"][0]
    assert issue["code"] == "unsupported_reference"
    assert issue["pointer"] == "/paths/~1p/get/parameters/0"


def test_circular_reference_chain_is_422():
    doc = make_doc(
        {
            "/p": {"get": op(parameters=[
                {"$ref": "#/components/parameters/A"},
            ])},
        },
        components={"parameters": {
            "A": {"$ref": "#/components/parameters/B"},
            "B": {"$ref": "#/components/parameters/A"},
        }},
    )
    resp = materialize(doc, "GET /p")
    assert resp.status_code == 422
    assert resp.json()["issues"][0]["code"] == "circular_reference"


def test_path_level_reference_is_resolved_for_operation_lookup():
    doc = make_doc(
        {
            "/p": {
                "$ref": "#/components/pathItems/P",
            },
        },
        components={"pathItems": {
            "P": {"get": op(parameters=[
                param("q", "query"),
            ])},
        }},
    )
    resp = materialize(doc, "GET /p", {"query": {"q": "1"}})
    assert resp.status_code == 200, resp.text
    assert resp.json()["request_target"] == "/p?q=1"


def test_dangling_path_level_reference_is_422():
    doc = make_doc({
        "/p": {"$ref": "#/components/pathItems/Missing"},
    })
    resp = materialize(doc, "GET /p")
    assert resp.status_code == 422
    assert resp.json()["issues"][0]["code"] == "unsupported_reference"


# --------------------------------------------------------------------------- #
# 422: missing / unknown / duplicate / invalid
# --------------------------------------------------------------------------- #

def test_missing_required_path_value_is_422():
    doc = make_doc({"/p/{id}": {"get": op(parameters=[
        param("id", "path", required=True),
    ])}})
    resp = materialize(doc, "GET /p/{id}", {})
    assert resp.status_code == 422
    issue = resp.json()["issues"][0]
    assert issue["code"] == "missing_required_value"
    assert issue["in"] == "path"
    assert issue["name"] == "id"
    assert issue["pointer"].startswith("/paths/")
    assert issue["pointer"].endswith("/get/parameters/0")


def test_path_parameters_are_required_even_without_required_flag():
    doc = make_doc({"/p/{id}": {"get": op(parameters=[
        param("id", "path"),
    ])}})
    resp = materialize(doc, "GET /p/{id}", {})
    assert resp.status_code == 422
    assert resp.json()["issues"][0]["code"] == "missing_required_value"


def test_missing_required_query_value_is_422():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("q", "query", required=True),
    ])}})
    resp = materialize(doc, "GET /p", {})
    assert resp.status_code == 422
    assert resp.json()["issues"][0]["code"] == "missing_required_value"


def test_omitted_optional_parameter_simply_disappears():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("q", "query"),
        param("X-T", "header"),
        param("s", "cookie"),
    ])}})
    resp = materialize(doc, "GET /p", {})
    assert resp.status_code == 200
    assert resp.json() == {"request_target": "/p", "headers": {}, "cookie": ""}


def test_unknown_supplied_value_is_422_with_value_pointer():
    doc = make_doc({"/p": {"get": op()}})
    resp = materialize(doc, "GET /p", {"query": {"ghost": "x"}})
    assert resp.status_code == 422
    issue = resp.json()["issues"][0]
    assert issue["code"] == "unknown_value"
    assert issue["in"] == "query"
    assert issue["name"] == "ghost"
    assert issue["value_pointer"] == "/values/query/ghost"


def test_unknown_value_in_each_location_is_422():
    doc = make_doc({"/p": {"get": op()}})
    for location in ("path", "query", "header", "cookie"):
        resp = materialize(doc, "GET /p", {location: {"ghost": 1}})
        assert resp.status_code == 422, location
        assert resp.json()["issues"][0]["in"] == location


def test_duplicate_parameter_within_operation_list_is_422():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("q", "query"),
        param("q", "query"),
    ])}})
    resp = materialize(doc, "GET /p", {"query": {"q": "1"}})
    assert resp.status_code == 422
    issue = resp.json()["issues"][0]
    assert issue["code"] == "duplicate_parameter"
    assert issue["pointer"] == "/paths/~1p/get/parameters/1"


def test_duplicate_parameter_within_path_level_list_is_422():
    doc = make_doc({"/p": {
        "parameters": [param("q", "query"), param("q", "query")],
        "get": op(),
    }})
    resp = materialize(doc, "GET /p", {"query": {"q": "1"}})
    assert resp.status_code == 422
    issue = resp.json()["issues"][0]
    assert issue["code"] == "duplicate_parameter"
    assert issue["pointer"] == "/paths/~1p/parameters/1"


def test_same_in_name_on_path_and_operation_is_override_not_duplicate():
    doc = make_doc({"/p": {
        "parameters": [param("q", "query")],
        "get": op(parameters=[param("q", "query")]),
    }})
    resp = materialize(doc, "GET /p", {"query": {"q": "1"}})
    assert resp.status_code == 200, resp.text


def test_same_name_different_location_is_not_a_duplicate():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("x", "query"), param("x", "header"),
    ])}})
    resp = materialize(doc, "GET /p",
                       {"query": {"x": "1"}, "header": {"x": "2"}})
    assert resp.status_code == 200, resp.text


def test_null_value_is_422():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("q", "query"),
    ])}})
    resp = materialize(doc, "GET /p", {"query": {"q": None}})
    assert resp.status_code == 422
    assert resp.json()["issues"][0]["code"] == "unsupported_value_type"


def test_deep_object_requires_object_value():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("f", "query", style="deepObject"),
    ])}})
    resp = materialize(doc, "GET /p", {"query": {"f": ["a", "b"]}})
    assert resp.status_code == 422
    assert resp.json()["issues"][0]["code"] == "unsupported_value_type"


def test_deep_object_rejects_nested_members():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("f", "query", style="deepObject"),
    ])}})
    resp = materialize(doc, "GET /p",
                       {"query": {"f": {"a": {"b": 1}}}})
    assert resp.status_code == 422
    issue = resp.json()["issues"][0]
    assert issue["code"] == "unsupported_value_type"
    assert issue["value_pointer"] == "/values/query/f/a"


def test_space_delimited_requires_array():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("t", "query", style="spaceDelimited"),
    ])}})
    resp = materialize(doc, "GET /p", {"query": {"t": "hello"}})
    assert resp.status_code == 422
    assert resp.json()["issues"][0]["code"] == "unsupported_value_type"


def test_pipe_delimited_rejects_objects():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("t", "query", style="pipeDelimited"),
    ])}})
    resp = materialize(doc, "GET /p", {"query": {"t": {"a": "1"}}})
    assert resp.status_code == 422
    assert resp.json()["issues"][0]["code"] == "unsupported_value_type"


def test_nested_array_member_is_422_with_member_pointer():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("t", "query", schema={"type": "array"}),
    ])}})
    resp = materialize(doc, "GET /p",
                       {"query": {"t": ["a", ["b", "c"]]}})
    assert resp.status_code == 422
    issue = resp.json()["issues"][0]
    assert issue["code"] == "unsupported_value_type"
    assert issue["value_pointer"] == "/values/query/t/1"


def test_explode_true_rejected_for_space_delimited():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("t", "query", style="spaceDelimited", explode=True),
    ])}})
    resp = materialize(doc, "GET /p", {"query": {"t": ["a"]}})
    assert resp.status_code == 422
    assert resp.json()["issues"][0]["code"] == "invalid_parameter"


def test_deep_object_requires_explode_true():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("f", "query", style="deepObject", explode=False),
    ])}})
    resp = materialize(doc, "GET /p", {"query": {"f": {"a": "1"}}})
    assert resp.status_code == 422
    assert resp.json()["issues"][0]["code"] == "invalid_parameter"


def test_unknown_style_for_location_is_422():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("x", "header", style="label"),
    ])}})
    resp = materialize(doc, "GET /p", {"header": {"x": "1"}})
    assert resp.status_code == 422
    assert resp.json()["issues"][0]["code"] == "invalid_parameter"


def test_cookie_only_supports_form_style():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("s", "cookie", style="simple"),
    ])}})
    resp = materialize(doc, "GET /p", {"cookie": {"s": "1"}})
    assert resp.status_code == 422
    assert resp.json()["issues"][0]["code"] == "invalid_parameter"


def test_allow_reserved_on_non_query_parameter_is_422():
    doc = make_doc({"/p/{id}": {"get": op(parameters=[
        param("id", "path", required=True, allowReserved=True),
    ])}})
    resp = materialize(doc, "GET /p/{id}", {"path": {"id": "a"}})
    assert resp.status_code == 422
    assert resp.json()["issues"][0]["code"] == "invalid_parameter"


def test_parameter_without_in_is_422():
    doc = make_doc({"/p": {"get": op(parameters=[{"name": "q"}])}})
    resp = materialize(doc, "GET /p")
    assert resp.status_code == 422
    issue = resp.json()["issues"][0]
    assert issue["code"] == "invalid_parameter"
    assert issue["pointer"] == "/paths/~1p/get/parameters/0"


def test_path_placeholder_without_declared_parameter_is_422():
    doc = make_doc({"/p/{ghost}": {"get": op()}})
    resp = materialize(doc, "GET /p/{ghost}")
    assert resp.status_code == 422
    issue = resp.json()["issues"][0]
    assert issue["code"] == "unresolved_path_template"
    assert issue["name"] == "ghost"
    assert issue["pointer"] == "/paths/~1p~1{ghost}"


def test_header_control_character_is_422():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("X-T", "header"),
    ])}})
    resp = materialize(doc, "GET /p", {"header": {"X-T": "ab\ncd"}})
    assert resp.status_code == 422
    assert resp.json()["issues"][0]["code"] == "unsupported_value_type"


def test_issues_are_sorted_stably():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("b", "query", required=True),
        param("a", "query", required=True),
    ])}})
    resp = materialize(
        doc, "GET /p",
        {"query": {"zz": "1", "aa": "2"}},
    )
    assert resp.status_code == 422
    codes = [(i["code"], i.get("name")) for i in resp.json()["issues"]]
    assert codes == sorted(codes)
    # missing required (a, b) sort before unknown (aa, zz)
    names = [i.get("name") for i in resp.json()["issues"]]
    assert names[:2] == ["a", "b"]
    assert names[2:] == ["aa", "zz"]


def test_multiple_errors_are_all_reported():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("q", "query", required=True),
    ])}})
    resp = materialize(doc, "GET /p", {"query": {"ghost": "x"}})
    assert resp.status_code == 422
    codes = {i["code"] for i in resp.json()["issues"]}
    assert codes == {"missing_required_value", "unknown_value"}


# --------------------------------------------------------------------------- #
# 400: request shape
# --------------------------------------------------------------------------- #

def test_non_openapi_document_is_400():
    resp = materialize({"foo": "bar"}, "GET /p")
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_materialize_request"


def test_malformed_selector_is_400():
    doc = make_doc({"/p": {"get": op()}})
    for bad in ("GET", "garbage", "GET/p", "BREW /p"):
        assert materialize(doc, bad).status_code == 400, bad


def test_unknown_path_is_400_not_422():
    doc = make_doc({"/p": {"get": op()}})
    resp = materialize(doc, "GET /nope")
    assert resp.status_code == 400
    assert "unknown operation" in resp.json()["messages"][0]


def test_unknown_method_is_400():
    doc = make_doc({"/p": {"get": op()}})
    resp = materialize(doc, "POST /p")
    assert resp.status_code == 400


def test_unknown_values_group_is_400():
    doc = make_doc({"/p": {"get": op()}})
    resp = materialize(doc, "GET /p", {"body": {"x": 1}})
    assert resp.status_code == 400


def test_values_group_that_is_not_an_object_is_400():
    doc = make_doc({"/p": {"get": op()}})
    resp = materialize(doc, "GET /p", {"query": ["a"]})
    assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# Stability / non-mutation
# --------------------------------------------------------------------------- #

def test_input_document_and_values_are_not_mutated():
    doc = make_doc(
        {
            "/p/{id}": {"get": op(parameters=[
                {"$ref": "#/components/parameters/Id"},
                param("q", "query"),
            ])},
        },
        components={"parameters": {
            "Id": param("id", "path", required=True),
        }},
    )
    values = {
        "path": {"id": "x/y"},
        "query": {"q": {"b": "2", "a": "1"}},
    }
    doc_before = copy.deepcopy(doc)
    values_before = copy.deepcopy(values)
    resp = materialize(doc, "GET /p/{id}", values)
    assert resp.status_code == 200, resp.text
    assert doc == doc_before
    assert values == values_before


def test_output_is_stable_across_input_key_orderings():
    def build(values):
        doc = make_doc({"/p": {"get": op(parameters=[
            param("z", "query", style="deepObject"),
            param("a", "query", schema={"type": "array"}),
            param("X-M", "header"),
            param("s2", "cookie"), param("s1", "cookie"),
        ])}})
        return materialize(doc, "GET /p", values).json()

    first = build({
        "query": {"z": {"y": "1", "x": "2"}, "a": ["3", "1", "2"]},
        "header": {"X-M": "v"},
        "cookie": {"s2": "b", "s1": "a"},
    })
    second = build({
        "cookie": {"s1": "a", "s2": "b"},
        "header": {"X-M": "v"},
        "query": {"a": ["3", "1", "2"], "z": {"x": "2", "y": "1"}},
    })
    assert first == second
    assert list(first["headers"]) == ["X-M"]
    assert first["cookie"] == "s1=a; s2=b"


def test_array_member_order_is_preserved():
    doc = make_doc({"/p": {"get": op(parameters=[
        param("t", "query", schema={"type": "array"}),
    ])}})
    resp = materialize(doc, "GET /p",
                       {"query": {"t": ["z", "a", "m"]}})
    assert resp.json()["request_target"] == "/p?t=z&t=a&t=m"


def test_object_members_render_in_sorted_key_order():
    doc = make_doc({"/p/{o}": {"get": op(parameters=[
        param("o", "path", required=True, style="label", explode=True),
    ])}})
    resp = materialize(doc, "GET /p/{o}",
                       {"path": {"o": {"d": "4", "b": "2", "a": "1", "c": "3"}}})
    assert resp.json()["request_target"] == "/p/.a=1,b=2,c=3,d=4"


# --------------------------------------------------------------------------- #
# OpenAPI 3.1 sibling annotations on a referenced parameter
# --------------------------------------------------------------------------- #

def test_ref_sibling_annotation_overrides_referenced_member():
    doc = make_doc(
        {
            "/p": {"get": op(parameters=[
                {"$ref": "#/components/parameters/Q",
                 "style": "pipeDelimited"},
            ])},
        },
        components={"parameters": {
            "Q": param("q", "query", style="form",
                       schema={"type": "array"}),
        }},
    )
    resp = materialize(doc, "GET /p", {"query": {"q": ["a", "b"]}})
    assert resp.status_code == 200, resp.text
    assert resp.json()["request_target"] == "/p?q=a|b"
