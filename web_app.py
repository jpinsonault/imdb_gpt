from flask import Flask, request, jsonify, render_template
import logging
import sys
from pathlib import Path

# Add scripts to path so imports work
sys.path.append(str(Path(__file__).parent))

from scripts.inference import MovieSearchEngine

app = Flask(__name__)

# Initialize engine globally
print("Loading Search Engine... please wait.")
engine = MovieSearchEngine()
print("Search Engine Loaded.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search_api():
    data = request.json
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400
    
    query = data.get("query", "")
    top_k = data.get("top_k", 50)
    
    # Simple parsing: Put the raw query into primaryTitle
    # We could extend this to parse "Matrix 1999" into Title="Matrix", Year=1999
    search_params = {
        "primaryTitle": query
    }
    
    # Support explicit field overrides if sent in JSON
    # e.g. {"query": "Matrix", "filters": {"startYear": 1999}}
    filters = data.get("filters", {})
    search_params.update(filters)
    
    try:
        results = engine.search(search_params, top_k=top_k)
        return jsonify({
            "results": results, 
            "count": len(results),
            "interpreted_query": search_params
        })
    except Exception as e:
        logging.error(f"Search error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)