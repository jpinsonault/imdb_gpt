import importlib
import sys

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def client():
    """Create a Flask test client with a mocked search engine."""
    mock_engine = MagicMock()
    mock_engine.search_movies.return_value = [
        {"primaryTitle": "Test Movie", "startYear": 2020, "averageRating": 8.0, "genres": "Drama"},
    ]
    mock_engine.search_people.return_value = [
        {"primaryName": "Test Person", "birthYear": 1980},
    ]
    mock_engine.get_movie_detail.return_value = {
        "primaryTitle": "Test Movie",
        "tconst": "tt0000001",
        "startYear": 2020,
        "genres": ["Drama"],
    }
    mock_engine.get_person_detail.return_value = {
        "primaryName": "Test Person",
        "nconst": "nm0000001",
        "birthYear": 1980,
    }

    # Replace the inference module so web_app doesn't load the real engine
    inference_mock = MagicMock()
    inference_mock.HybridSearchEngine.return_value = mock_engine
    saved = sys.modules.get("scripts.inference")
    sys.modules["scripts.inference"] = inference_mock

    try:
        import web_app
        importlib.reload(web_app)
        web_app.app.testing = True
        with web_app.app.test_client() as c:
            yield c
    finally:
        if saved is not None:
            sys.modules["scripts.inference"] = saved
        else:
            sys.modules.pop("scripts.inference", None)


def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"html" in resp.data.lower() or b"<!doctype" in resp.data.lower()


def test_search_movies_api(client):
    resp = client.post("/api/search", json={"query": "test", "search_type": "movie"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "results" in data
    assert "count" in data
    assert data["count"] >= 1


def test_search_people_api(client):
    resp = client.post("/api/search", json={"query": "test", "search_type": "person"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "results" in data
    assert "count" in data


def test_movie_detail_api(client):
    resp = client.get("/api/movie/0")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "primaryTitle" in data
    assert "tconst" in data


def test_person_detail_api(client):
    resp = client.get("/api/person/0")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "primaryName" in data
    assert "nconst" in data
