from flask import Flask, request, jsonify, render_template
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from scripts.inference import HybridSearchEngine

app = Flask(__name__)

print("Loading Search Engine... please wait.")
engine = HybridSearchEngine()
print("Search Engine Loaded.")


@app.route("/")
def index():
    return render_template("index.html")


def _passes_filters(row, filters):
    if not filters:
        return True

    start_year = filters.get("startYear")
    if start_year:
        sy = str(row.get("startYear", "")).strip()
        if not sy.startswith(str(start_year).strip()):
            return False

    min_rating = filters.get("averageRating")
    if min_rating:
        try:
            threshold = float(min_rating)
            val = float(str(row.get("averageRating", "0")).strip() or 0.0)
        except ValueError:
            val = 0.0
        if val < threshold:
            return False

    genre_filter = filters.get("genres")
    if genre_filter:
        gf = str(genre_filter).strip().lower()
        if gf:
            genres_str = str(row.get("genres", "")).lower()
            if gf not in genres_str:
                return False

    return True


@app.route("/api/search", methods=["POST"])
def search_api():
    data = request.json or {}
    query = data.get("query", "") or ""
    top_k = int(data.get("top_k", 50) or 50)
    filters = data.get("filters", {}) or {}
    search_type = (data.get("search_type") or "movie").strip().lower()

    try:
        if search_type == "person":
            raw_results = engine.search_people(query, top_k=top_k)
            filtered_results = raw_results
        else:
            raw_results = engine.search_movies(query, top_k=top_k)
            filtered_results = [r for r in raw_results if _passes_filters(r, filters)]

        return jsonify(
            {
                "results": filtered_results,
                "count": len(filtered_results),
                "interpreted_query": {
                    "primaryTitle": query.strip(),
                    "filters": filters,
                    "search_type": search_type,
                },
            }
        )
    except Exception as e:
        logging.error(f"Search error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/movie/<int:movie_id>")
def movie_page(movie_id: int):
    try:
        detail = engine.get_movie_detail(movie_id)
    except KeyError:
        return "Movie not found", 404
    except Exception as e:
        logging.error(f"Movie detail error: {e}", exc_info=True)
        return "Internal error", 500
    return render_template("movie.html", movie=detail)


@app.route("/person/<int:person_id>")
def person_page(person_id: int):
    try:
        detail = engine.get_person_detail(person_id)
    except KeyError:
        return "Person not found", 404
    except Exception as e:
        logging.error(f"Person detail error: {e}", exc_info=True)
        return "Internal error", 500
    return render_template("person.html", person=detail)


@app.route("/api/movie/<int:movie_id>")
def movie_detail_api(movie_id: int):
    try:
        detail = engine.get_movie_detail(movie_id)
        return jsonify(detail)
    except KeyError:
        return jsonify({"error": "Movie not found"}), 404
    except Exception as e:
        logging.error(f"Movie detail API error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/person/<int:person_id>")
def person_detail_api(person_id: int):
    try:
        detail = engine.get_person_detail(person_id)
        return jsonify(detail)
    except KeyError:
        return jsonify({"error": "Person not found"}), 404
    except Exception as e:
        logging.error(f"Person detail API error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
