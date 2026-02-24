import pytest
import torch

from scripts.autoencoder.fields import NumericDigitCategoryField, TextField
from scripts.simple_set.model import HybridSetModel


@pytest.fixture
def tiny_model():
    """Build a minimal HybridSetModel for shape testing."""
    # Movie fields
    title_field = TextField("primaryTitle", max_length=16, base_size=8)
    year_field = NumericDigitCategoryField("startYear")

    for t in ["Jaws", "Star Wars", "Alien", "Matrix", "Blade"]:
        title_field.accumulate_stats(t)
    for y in [1975, 1977, 1979, 1999, 2010]:
        year_field.accumulate_stats(y)
    title_field.finalize_stats()
    year_field.finalize_stats()

    movie_fields = [title_field, year_field]

    # Person fields
    name_field = TextField("primaryName", max_length=16, base_size=8)
    birth_field = NumericDigitCategoryField("birthYear")

    for n in ["Spielberg", "Lucas", "Weaver"]:
        name_field.accumulate_stats(n)
    for y in [1946, 1944, 1949]:
        birth_field.accumulate_stats(y)
    name_field.finalize_stats()
    birth_field.finalize_stats()

    person_fields = [name_field, birth_field]

    num_movies = 5
    num_people = 3
    dim = 8

    heads_config = {"cast": {"weight": 1.0}}

    model = HybridSetModel(
        movie_fields=movie_fields,
        person_fields=person_fields,
        num_movies=num_movies,
        num_people=num_people,
        heads_config=heads_config,
        movie_head_vocab_sizes={"cast": num_people},
        movie_head_local_to_global={"cast": torch.arange(num_people)},
        person_head_vocab_sizes={"cast": num_movies},
        person_head_local_to_global={"cast": torch.arange(num_movies)},
        movie_dim=dim,
        hidden_dim=16,
        person_dim=dim,
        dropout=0.0,
        logit_scale=1.0,
        film_bottleneck_dim=4,
        decoder_num_layers=1,
        decoder_num_heads=2,
        decoder_ff_multiplier=2,
        decoder_dropout=0.0,
        decoder_norm_first=False,
    )

    return model, movie_fields, person_fields, num_movies, num_people, dim


def test_forward_movie_only(tiny_model):
    model, movie_fields, _, _, num_people, dim = tiny_model
    model.train()

    idx = torch.tensor([0, 1])
    out = model(movie_indices=idx)

    assert "movie" in out
    side = out["movie"]
    assert side.embeddings.shape == (2, dim)
    assert "cast" in side.logits_dict
    assert side.logits_dict["cast"].shape == (2, num_people)
    assert len(side.recon_table) == len(movie_fields)


def test_forward_person_only(tiny_model):
    model, _, person_fields, num_movies, _, dim = tiny_model
    model.train()

    idx = torch.tensor([0, 1, 2])
    out = model(person_indices=idx)

    assert "person" in out
    side = out["person"]
    assert side.embeddings.shape == (3, dim)
    assert "cast" in side.logits_dict
    assert side.logits_dict["cast"].shape == (3, num_movies)
    assert len(side.recon_table) == len(person_fields)


def test_forward_both_sides(tiny_model):
    model, *_ = tiny_model
    model.train()

    out = model(movie_indices=torch.tensor([0, 1]), person_indices=torch.tensor([0]))

    assert "movie" in out
    assert "person" in out
    assert "film_reg" in out


def test_eval_no_film_reg(tiny_model):
    model, *_ = tiny_model
    model.eval()

    with torch.no_grad():
        out = model(movie_indices=torch.tensor([0]), person_indices=torch.tensor([0]))

    assert "film_reg" not in out


def test_recon_table_shapes(tiny_model):
    model, movie_fields, person_fields, _, _, dim = tiny_model
    model.eval()

    batch = 2
    with torch.no_grad():
        out = model(movie_indices=torch.arange(batch), person_indices=torch.arange(batch))

    for i, tensor in enumerate(out["movie"].recon_table):
        assert tensor.shape[0] == batch, f"Movie recon field {i} wrong batch dim"

    for i, tensor in enumerate(out["person"].recon_table):
        assert tensor.shape[0] == batch, f"Person recon field {i} wrong batch dim"


# ---------------------------------------------------------------------------
# FiLM regularization
# ---------------------------------------------------------------------------

def test_film_reg_is_positive_scalar(tiny_model):
    model, *_ = tiny_model
    model.train()

    out = model(movie_indices=torch.tensor([0, 1]), person_indices=torch.tensor([0]))
    reg = out["film_reg"]
    assert reg.ndim == 0
    assert reg.item() >= 0.0


def test_film_reg_absent_when_scale_zero(tiny_model):
    model, *_ = tiny_model
    model.train()

    out = model(movie_indices=torch.tensor([0]), person_indices=torch.tensor([0]),
                film_scale=0.0)
    assert "film_reg" not in out


# ---------------------------------------------------------------------------
# Search encoder loss
# ---------------------------------------------------------------------------

def test_search_encoder_loss_scalar(tiny_model):
    model, movie_fields, person_fields, *_ = tiny_model
    model.train()

    # Get title tokens for a sample
    title_field = movie_fields[0]
    name_field = person_fields[0]
    title_tokens = title_field.transform("Jaws").unsqueeze(0)    # (1, L)
    name_tokens = name_field.transform("Spielberg").unsqueeze(0)  # (1, L)

    loss = model.compute_search_encoder_loss(
        movie_indices=torch.tensor([0]),
        movie_title_tokens=title_tokens,
        person_indices=torch.tensor([0]),
        person_name_tokens=name_tokens,
    )
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0


def test_search_encoder_embedding_gradients_stopped(tiny_model):
    """Embedding table should NOT receive gradients from search encoder loss."""
    model, movie_fields, *_ = tiny_model
    model.train()

    title_field = movie_fields[0]
    title_tokens = title_field.transform("Jaws").unsqueeze(0)

    loss = model.compute_search_encoder_loss(
        movie_indices=torch.tensor([0]),
        movie_title_tokens=title_tokens,
        person_indices=None,
        person_name_tokens=None,
    )
    loss.backward()

    # Embedding table should have no gradient (stop-grad in compute_search_encoder_loss)
    assert model.movie_embeddings.weight.grad is None
