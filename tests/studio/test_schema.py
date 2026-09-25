"""The model form edits hidden layers as rows, not as two separate fields."""

from __future__ import annotations

from teotensor.nn import MLPClassifier
from teotensor.nn.activations import ReLU, Tanh
from teotensor.studio.page import PAGE_HTML
from teotensor.studio.schema import apply_fields, form_fields


def test_field_titles_are_words() -> None:
    assert 'weight_init: {label: "Weight fill"}' in PAGE_HTML
    assert 'bias_init: {label: "Bias fill"}' in PAGE_HTML


def test_form_leaves_bias_at_its_default() -> None:
    keys = {field["key"] for field in form_fields(MLPClassifier())}
    assert "bias" not in keys
    assert MLPClassifier().bias is True


def test_default_mlp_is_one_hidden_row() -> None:
    fields = form_fields(MLPClassifier())
    layers = next(field for field in fields if field["kind"] == "layers")
    assert layers["value"] == [{"neurons": 100, "activation": "ReLU"}]
    keys = {field["key"] for field in fields}
    assert "hidden_layer_sizes" not in keys
    assert "hidden_activations" not in keys
    assert "activation" not in keys


def test_mixed_rows_round_trip() -> None:
    model = MLPClassifier()
    apply_fields(
        model,
        [
            {
                "key": "layers",
                "kind": "layers",
                "value": [
                    {"neurons": 32, "activation": "Tanh"},
                    {"neurons": 16, "activation": "ReLU"},
                ],
            }
        ],
    )
    assert model.hidden_layer_sizes == (32, 16)
    assert isinstance(model.hidden_activations[0], Tanh)
    assert isinstance(model.hidden_activations[1], ReLU)
    again = next(field for field in form_fields(model) if field["kind"] == "layers")
    assert again["value"] == [
        {"neurons": 32, "activation": "Tanh"},
        {"neurons": 16, "activation": "ReLU"},
    ]


def test_one_activation_stays_the_shared_default() -> None:
    model = MLPClassifier()
    apply_fields(
        model,
        [
            {
                "key": "layers",
                "kind": "layers",
                "value": [{"neurons": 8, "activation": "ReLU"}],
            }
        ],
    )
    assert model.hidden_layer_sizes == (8,)
    assert model.hidden_activations is None
    assert isinstance(model.activation, ReLU)


def test_no_rows_is_a_direct_map() -> None:
    model = MLPClassifier()
    apply_fields(model, [{"key": "layers", "kind": "layers", "value": []}])
    assert model.hidden_layer_sizes == ()
    assert model.hidden_activations is None
    assert model.activation is None
