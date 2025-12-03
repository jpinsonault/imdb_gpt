from flask import Flask, request, jsonify, render_template
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from scripts.inference import MovieSearchEngine

app = Flask(__name__)

print("Loading Search Engine... please wait.")
engine = MovieSearchEngine()
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
    data = request.json
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400

    query = data.get("query", "") or ""
    top_k = int(data.get("top_k", 50) or 50)
    filters = data.get("filters", {}) or {}

    try:
        raw_results = engine.search_by_title(query, top_k=top_k)
        filtered_results = [r for r in raw_results if _passes_filters(r, filters)]

        return jsonify(
            {
                "results": filtered_results,
                "count": len(filtered_results),
                "interpreted_query": {
                    "primaryTitle": query.strip(),
                    "filters": filters,
                },
            }
        )
    except Exception as e:
        logging.error(f"Search error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
