import sqlite3
import numpy as np
import logging # Added for potential warnings/errors
from flask import Flask, jsonify, request, render_template

# Assuming scripts.autoencoder paths are correct relative to where app.py is run
from scripts.autoencoder.imdb_row_autoencoders import PeopleAutoencoder, TitlesAutoencoder

# --- Configuration ---
CONFIG = {
    'DB_PATH': 'data/imdb.db',
    'TITLE_MODEL_PATH': 'models/',
    'PEOPLE_MODEL_PATH': 'models/',
    'FAISS_INDEX_PATH': 'models/faiss.index',
    'LATENT_DIM': 32,
    'MAX_RESULTS_SIMILAR': 10,
}

# --- Flask App Initialization ---
app = Flask(__name__, static_folder="static")
app.config['SECRET_KEY'] = 'your_secret_key_here' # Good practice for Flask

# --- Initialize Autoencoders ---
try:
    title_encoder = TitlesAutoencoder({'latent_dim': CONFIG['LATENT_DIM']}, CONFIG['DB_PATH'], CONFIG['TITLE_MODEL_PATH'])
    people_encoder = PeopleAutoencoder({'latent_dim': CONFIG['LATENT_DIM']}, CONFIG['DB_PATH'], CONFIG['PEOPLE_MODEL_PATH'])

    # Accumulate stats and finalize (essential before loading models)
    print("Accumulating Title stats...")
    title_encoder.accumulate_stats()
    title_encoder.finalize_stats()
    print("Finished Title stats.")

    print("Accumulating People stats...")
    people_encoder.accumulate_stats()
    people_encoder.finalize_stats()
    print("Finished People stats.")


    # Load models after tokenizers/stats are finalized
    print("Loading models...")
    title_encoder.load_model()
    people_encoder.load_model()
    print("Models loaded.")

except FileNotFoundError as e:
    logging.error(f"Error loading models or stats: {e}. Make sure models and stats exist.")
    # Depending on your setup, you might want to exit or handle this differently
    # For now, we'll let it potentially fail later if models aren't loaded.
    title_encoder = None
    people_encoder = None
except Exception as e:
    logging.error(f"An unexpected error occurred during initialization: {e}", exc_info=True)
    title_encoder = None
    people_encoder = None


# --- Helper Function ---
def get_db_connection():
    """Establishes a database connection."""
    try:
        conn = sqlite3.connect(CONFIG['DB_PATH'])
        conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
        return conn
    except sqlite3.Error as e:
        logging.error(f"Database connection error: {e}")
        return None

# --- Routes ---
@app.route("/")
def index():
    """Serves the main search page."""
    return render_template("index.html")

@app.route("/api/search/<db>")
def search(db):
    """API endpoint for searching titles or people."""
    query = request.args.get("q", "")
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500
    cursor = conn.cursor()
    results = []

    try:
        if db == "titles":
            cursor.execute("""
                SELECT tconst, primaryTitle FROM titles
                WHERE primaryTitle LIKE ? AND titleType IN ('movie', 'tvSeries', 'tvMovie', 'tvMiniSeries')
                ORDER BY numVotes DESC
                LIMIT 20
            """, ('%' + query + '%',))
        elif db == "people":
             cursor.execute("""
                SELECT nconst, primaryName FROM people
                WHERE primaryName LIKE ?
                LIMIT 20
            """, ('%' + query + '%',))
        else:
            conn.close()
            return jsonify({"error": "Invalid database type"}), 400

        results = [dict(row) for row in cursor.fetchall()] # Convert to list of dicts
    except sqlite3.Error as e:
        logging.error(f"Database query error in /api/search: {e}")
        return jsonify({"error": "Database query failed"}), 500
    finally:
        conn.close()

    # Convert results to the list of lists format expected by the old frontend if needed
    # Or update frontend to handle list of dicts
    results_list_of_lists = [[row['tconst'] if 'tconst' in row else row['nconst'],
                              row['primaryTitle'] if 'primaryTitle' in row else row['primaryName']]
                             for row in results]

    return jsonify(results_list_of_lists)


@app.route("/inspect/<db>/<item_id>")
def inspect_page(db, item_id):
    """Serves the inspection page for a specific item."""
    # Pass db and item_id to the template for the initial API call
    return render_template("inspect.html", db=db, item_id=item_id)


@app.route("/api/initial_data/<db>/<item_id>")
def get_initial_data(db, item_id):
    """API endpoint to fetch initial data for the inspection page."""
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500
    cursor = conn.cursor()
    row_data = None
    latent_vector = None
    encoder_instance = None

    try:
        if db == "titles" and title_encoder:
            encoder_instance = title_encoder
            cursor.execute("""
                SELECT t.tconst, t.primaryTitle, t.startYear, t.runtimeMinutes,
                       t.averageRating, t.numVotes, GROUP_CONCAT(g.genre, ',') AS genres
                FROM titles t
                LEFT JOIN title_genres g ON t.tconst = g.tconst
                WHERE t.tconst = ?
                GROUP BY t.tconst
            """, (item_id,))
            data = cursor.fetchone()
            if data:
                row_data = {
                    "tconst": data["tconst"], # Include the ID
                    "primaryTitle": data["primaryTitle"],
                    "startYear": data["startYear"],
                    "runtimeMinutes": data["runtimeMinutes"],
                    "averageRating": data["averageRating"],
                    "numVotes": data["numVotes"],
                    # Split genres string into list, handle None case
                    "genres": data["genres"].split(',') if data["genres"] else []
                }

        elif db == "people" and people_encoder:
            encoder_instance = people_encoder
            cursor.execute("""
                SELECT p.nconst, p.primaryName, p.birthYear, GROUP_CONCAT(pp.profession, ',') AS professions
                FROM people p
                LEFT JOIN people_professions pp ON p.nconst = pp.nconst
                WHERE p.nconst = ?
                GROUP BY p.nconst
            """, (item_id,))
            data = cursor.fetchone()
            if data:
                row_data = {
                    "nconst": data["nconst"], # Include the ID
                    "primaryName": data["primaryName"],
                    "birthYear": data["birthYear"],
                    # Split professions string into list, handle None case
                    "professions": data["professions"].split(',') if data["professions"] else []
                }
        else:
            conn.close()
            if not title_encoder and db == "titles":
                 return jsonify({"error": "Title encoder not loaded"}), 500
            if not people_encoder and db == "people":
                return jsonify({"error": "People encoder not loaded"}), 500
            return jsonify({"error": "Invalid database type or encoder not loaded"}), 400

        if not data:
            conn.close()
            return jsonify({"error": "Item not found"}), 404

        # Encode the fetched row data
        if encoder_instance and row_data:
             # Transform the row data into the format expected by the encoder
            transformed_row = encoder_instance.transform_row(row_data)
            # Prepare input list for the encoder model, adding batch dimension
            encoder_inputs = [transformed_row[f.name][None,:] for f in encoder_instance.fields]
            # Predict the latent vector
            latent_vector = encoder_instance.encoder.predict(encoder_inputs)[0] # Get first item from batch

    except sqlite3.Error as e:
        logging.error(f"Database query error in /api/initial_data: {e}")
        return jsonify({"error": "Database query failed"}), 500
    except Exception as e:
         logging.error(f"Error during initial data encoding: {e}", exc_info=True)
         return jsonify({"error": f"Failed to encode initial data: {e}"}), 500
    finally:
        if conn:
            conn.close()

    if row_data and latent_vector is not None:
        return jsonify({
            "latent": latent_vector.tolist(),
            "fields": row_data # Send the original DB data
        })
    else:
        return jsonify({"error": "Failed to retrieve or encode data"}), 500


@app.route("/api/reconstruct/<db>", methods=["POST"])
def reconstruct_data(db):
    """API endpoint to encode and decode potentially modified data."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    latent_vector = None
    reconstructed_fields = None
    encoder_instance = None

    try:
        if db == "titles" and title_encoder:
            encoder_instance = title_encoder
        elif db == "people" and people_encoder:
            encoder_instance = people_encoder
        else:
            if not title_encoder and db == "titles":
                 return jsonify({"error": "Title encoder not loaded"}), 500
            if not people_encoder and db == "people":
                return jsonify({"error": "People encoder not loaded"}), 500
            return jsonify({"error": "Invalid database type or encoder not loaded"}), 400

        if encoder_instance:
            # 1. Transform the input data (potentially edited by user)
            # Ensure data types are correct (e.g., numeric fields are numbers)
            # This might require more robust type casting based on field definitions
            transformed_row = encoder_instance.transform_row(data) # Assumes transform_row handles types

            # 2. Prepare input list for the encoder model, adding batch dimension
            encoder_inputs = [transformed_row[f.name][None,:] for f in encoder_instance.fields]

            # 3. Encode to get the latent vector
            latent_vector_batch = encoder_instance.encoder.predict(encoder_inputs)
            latent_vector = latent_vector_batch[0] # Get first item from batch

            # 4. Reconstruct from the latent vector
            reconstructed_fields = encoder_instance.reconstruct_row(latent_vector)

    except Exception as e:
         logging.error(f"Error during reconstruction: {e}", exc_info=True)
         # Provide more specific error if possible
         return jsonify({"error": f"Failed to reconstruct data: {e}"}), 500

    if latent_vector is not None and reconstructed_fields is not None:
        return jsonify({
            "latent": latent_vector.tolist(),
            "reconstructed": reconstructed_fields
        })
    else:
        # Log the reason if possible (e.g., which step failed)
        return jsonify({"error": "Reconstruction process failed"}), 500


# --- Main Execution ---
if __name__ == "__main__":
    if not title_encoder or not people_encoder:
        print("\n*** Warning: One or more autoencoders failed to load. API endpoints might not work correctly. ***\n")
    # Set debug=False for production
    app.run(debug=True, port=5001)