import math

import pytest
import torch

from scripts.autoencoder.fields import (
    BooleanField,
    MultiCategoryField,
    NumericDigitCategoryField,
    ScalarField,
    Scaling,
    SingleCategoryField,
    TextField,
)


# ---------------------------------------------------------------------------
# Individual field roundtrip tests
# ---------------------------------------------------------------------------

def test_boolean_roundtrip():
    f = BooleanField("isAdult")
    f.accumulate_stats(1.0)
    f.accumulate_stats(0.0)
    f.finalize_stats()

    t = f.transform(1.0)
    assert t.shape == (1,)
    assert t.dtype == torch.float32
    assert f.render_ground_truth(t) == "True"

    t0 = f.transform(0.0)
    assert f.render_ground_truth(t0) == "False"


def test_scalar_none_roundtrip():
    f = ScalarField("votes", scaling=Scaling.NONE)
    for v in [100, 200, 300]:
        f.accumulate_stats(v)
    f.finalize_stats()

    t = f.transform(200)
    assert t.shape == (1,)
    assert t.dtype == torch.float32
    assert t.item() == pytest.approx(200.0)


def test_scalar_normalize_roundtrip():
    f = ScalarField("rating", scaling=Scaling.NORMALIZE)
    for v in range(11):  # 0..10
        f.accumulate_stats(float(v))
    f.finalize_stats()

    t = f.transform(5.0)
    assert t.shape == (1,)
    assert t.item() == pytest.approx(0.5)


def test_scalar_log_roundtrip():
    f = ScalarField("numVotes", scaling=Scaling.LOG)
    f.accumulate_stats(100)
    f.finalize_stats()

    t = f.transform(100)
    assert t.shape == (1,)
    assert t.item() == pytest.approx(math.log1p(100))


def test_numeric_digits_year():
    f = NumericDigitCategoryField("startYear")
    for y in [1975, 1977, 1979, 2010, 2024]:
        f.accumulate_stats(y)
    f.finalize_stats()

    t = f.transform(2024)
    assert t.dtype == torch.long
    assert f.render_ground_truth(t) == "2024"


def test_numeric_digits_rating_frac():
    f = NumericDigitCategoryField("averageRating", fraction_digits=1)
    for v in [5.0, 7.3, 8.1, 9.9]:
        f.accumulate_stats(v)
    f.finalize_stats()

    t = f.transform(7.3)
    assert t.dtype == torch.long
    assert f.render_ground_truth(t) == "7.3"


def test_numeric_digits_nan():
    f = NumericDigitCategoryField("endYear")
    f.accumulate_stats(2000)
    f.accumulate_stats(None)  # forces has_nan = True
    f.finalize_stats()

    t = f.transform(None)
    assert t.dtype == torch.long
    assert f.render_ground_truth(t) == "NaN"


def test_text_roundtrip():
    f = TextField("primaryTitle", max_length=16)
    for txt in ["Jaws", "Alien", "Star Wars"]:
        f.accumulate_stats(txt)
    f.finalize_stats()

    t = f.transform("Jaws")
    assert t.dtype == torch.long
    rendered = f.render_ground_truth(t)
    assert "Jaws" in rendered


def test_multi_category_roundtrip():
    f = MultiCategoryField("genres")
    f.accumulate_stats(["Action", "Drama"])
    f.accumulate_stats(["Comedy"])
    f.accumulate_stats(["Action", "Comedy"])
    f.finalize_stats()

    assert f.category_list == sorted(["Action", "Comedy", "Drama"])

    t = f.transform(["Action", "Drama"])
    assert t.dtype == torch.float32
    assert t.shape == (3,)  # 3 categories
    # Action and Drama bits set
    action_idx = f.category_list.index("Action")
    drama_idx = f.category_list.index("Drama")
    assert t[action_idx].item() == 1.0
    assert t[drama_idx].item() == 1.0


def test_single_category_roundtrip():
    f = SingleCategoryField("titleType")
    for cat in ["movie", "tvSeries", "tvMovie"]:
        f.accumulate_stats(cat)
    f.finalize_stats()

    t = f.transform("movie")
    assert t.dtype == torch.long
    assert t.shape == (1,)
    assert f.render_ground_truth(t) == "movie"


# ---------------------------------------------------------------------------
# Parametrized: state roundtrip
# ---------------------------------------------------------------------------

def _make_boolean():
    f = BooleanField("b")
    f.accumulate_stats(1.0)
    f.accumulate_stats(0.0)
    f.finalize_stats()
    return f, 1.0

def _make_scalar():
    f = ScalarField("s", scaling=Scaling.NORMALIZE)
    for v in range(11):
        f.accumulate_stats(float(v))
    f.finalize_stats()
    return f, 5.0

def _make_numeric():
    f = NumericDigitCategoryField("n")
    for v in [100, 200, 300]:
        f.accumulate_stats(v)
    f.finalize_stats()
    return f, 200

def _make_text():
    f = TextField("t", max_length=16)
    for txt in ["Hello", "World"]:
        f.accumulate_stats(txt)
    f.finalize_stats()
    return f, "Hello"

def _make_multi_cat():
    f = MultiCategoryField("mc")
    f.accumulate_stats(["A", "B"])
    f.accumulate_stats(["C"])
    f.finalize_stats()
    return f, ["A", "B"]

def _make_single_cat():
    f = SingleCategoryField("sc")
    for c in ["x", "y", "z"]:
        f.accumulate_stats(c)
    f.finalize_stats()
    return f, "y"


FIELD_FACTORIES = {
    "boolean": (_make_boolean, BooleanField),
    "scalar": (_make_scalar, ScalarField),
    "numeric": (_make_numeric, NumericDigitCategoryField),
    "text": (_make_text, TextField),
    "multi_cat": (_make_multi_cat, MultiCategoryField),
    "single_cat": (_make_single_cat, SingleCategoryField),
}


@pytest.mark.parametrize("field_id", FIELD_FACTORIES.keys())
def test_field_state_roundtrip(field_id):
    factory, cls = FIELD_FACTORIES[field_id]
    field, value = factory()

    tensor_a = field.transform(value)
    state = field.get_state()

    new_field = cls(field.name)
    new_field.set_state(state)

    tensor_b = new_field.transform(value)
    assert torch.equal(tensor_a, tensor_b), (
        f"State roundtrip mismatch for {field_id}: {tensor_a} vs {tensor_b}"
    )


# ---------------------------------------------------------------------------
# Parametrized: compute_loss runs without error
# ---------------------------------------------------------------------------

def _make_loss_tensors(field):
    """Create valid (pred, target) tensors for a field's compute_loss."""
    if isinstance(field, BooleanField):
        target = field.transform(1.0).unsqueeze(0)  # (1, 1)
        pred = torch.randn_like(target)
        return pred, target

    if isinstance(field, ScalarField):
        target = field.transform(5.0).unsqueeze(0)
        pred = torch.randn_like(target)
        return pred, target

    if isinstance(field, NumericDigitCategoryField):
        target = field.transform(200).unsqueeze(0)  # (1, P)
        P = target.shape[-1]
        V = field.vocab_size
        pred = torch.randn(1, P, V)
        return pred, target

    if isinstance(field, TextField):
        target = field.transform("Hello").unsqueeze(0)  # (1, L)
        L = target.shape[-1]
        V = field.tokenizer.get_vocab_size()
        pred = torch.randn(1, L, V)
        return pred, target

    if isinstance(field, MultiCategoryField):
        target = field.transform(["A", "B"]).unsqueeze(0)  # (1, C)
        pred = torch.randn_like(target)
        return pred, target

    if isinstance(field, SingleCategoryField):
        target = field.transform("y").unsqueeze(0)  # (1, 1)
        num_cats = len(field.category_list)
        pred = torch.randn(1, num_cats)
        return pred, target

    raise ValueError(f"Unknown field type: {type(field)}")


@pytest.mark.parametrize("field_id", FIELD_FACTORIES.keys())
def test_field_loss_runs(field_id):
    factory, _ = FIELD_FACTORIES[field_id]
    field, _ = factory()

    pred, target = _make_loss_tensors(field)
    loss = field.compute_loss(pred, target)

    assert loss.ndim == 0, "Loss should be scalar"
    assert not torch.isnan(loss), "Loss should not be NaN"
