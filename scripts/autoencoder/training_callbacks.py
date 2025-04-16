import logging
import random
import sqlite3
import numpy as np
from prettytable import PrettyTable, TableStyle
from typing import Any, List, Dict, Optional
import tensorflow as tf
from .row_autoencoder import RowAutoencoder
from .fields import BaseField, SPECIAL_PAD, SPECIAL_START, SPECIAL_END, NumericDigitCategoryField, TextField  # Import special tokens here
import os

class TensorBoardPerBatchLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, log_interval: int = 1):
        super().__init__()
        self.log_interval = log_interval
        self.file_writer = tf.summary.create_file_writer(log_dir)
        logging.info(f"filewriter:{self.file_writer}")

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        total_loss = logs.get("loss", 0.0)
        lr = self.model.optimizer.learning_rate
        lr_value = lr(self.model.optimizer.iterations) if callable(lr) else lr

        field_losses = {
            k.replace("_loss", ""): v 
            for k, v in logs.items() 
            if k.endswith("_loss") and k != "loss"
        }
        step = self.model.optimizer.iterations

        with self.file_writer.as_default():
            tf.summary.scalar("loss/total", total_loss, step=step)
            tf.summary.scalar("learning_rate", float(lr_value), step=step)
            for field, loss in field_losses.items():
                tf.summary.scalar(f"loss/{field}", loss, step=step)
            self.file_writer.flush()

        if (batch + 1) % self.log_interval == 0:
            field_loss_str = ", ".join([f"{field}: {loss:.4f}" for field, loss in field_losses.items()])
            message = (
                f"Batch {batch + 1} | Total Loss: {total_loss:.4f} | "
                f"LR: {float(lr_value):.6f} | Field Losses: {field_loss_str}"
            )
            logging.info(message)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with self.file_writer.as_default():
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    tf.summary.scalar(key, value, step=epoch)
            self.file_writer.flush()
        logging.info(f"Epoch {epoch + 1} ended. Details: {logs}")


class ModelSaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, row_autoencoder: RowAutoencoder, output_dir):
        super().__init__()
        self.row_autoencoder = row_autoencoder
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        self.row_autoencoder.save_model()
        print(f"Model saved to {self.output_dir} at the end of epoch {epoch + 1}")


class ReconstructionCallback(tf.keras.callbacks.Callback):
    def __init__(self, interval_batches, row_autoencoder: 'RowAutoencoder', db_path, num_samples=5):
        super().__init__()
        self.interval_batches = interval_batches
        self.row_autoencoder = row_autoencoder
        self.db_path = db_path  # kept for consistency
        self.num_samples = num_samples

        if not self.row_autoencoder.stats_accumulated:
            print("Warning: Stats not accumulated before initializing ReconstructionCallback. Accumulating now.")
            self.row_autoencoder.accumulate_stats()

        all_rows = list(row_autoencoder.row_generator())
        actual_num_samples = min(num_samples, len(all_rows))
        if actual_num_samples > 0:
            self.samples = random.sample(all_rows, actual_num_samples)
        else:
            self.samples = []
            print("Warning: No rows found for sampling in ReconstructionCallback.")

    def _tensor_to_string(self, field, main_tensor: np.ndarray, flag_tensor: Optional[np.ndarray] = None) -> str:
        try:
            if hasattr(field, "tokenizer") and field.tokenizer is not None:
                if main_tensor.ndim >= 2 and main_tensor.shape[-1] == field.tokenizer.get_vocab_size():
                    main_tensor = np.argmax(main_tensor, axis=-1)
            elif hasattr(field, "base") and hasattr(field, "fraction_digits"):
                if main_tensor.ndim >= 2:
                    main_tensor = np.argmax(main_tensor, axis=-1)
            if main_tensor.ndim > 1:
                main_tensor = main_tensor.flatten()
            if flag_tensor is not None:
                if flag_tensor.ndim > 1:
                    flag_tensor = flag_tensor.flatten()
                if flag_tensor.size > 1:
                    flag_tensor = flag_tensor[0:1]
                elif flag_tensor.size == 0:
                    flag_tensor = None
            return field.to_string(main_tensor, flag_tensor)
        except Exception as e:
            logging.warning(f"Callback: Error converting tensor to string for field {field.name}: {e}", exc_info=False)
            return "[Conversion Error]"

    def _top5_predictions(self, field, pred_vector: np.ndarray) -> str:
        if not np.isclose(np.sum(pred_vector), 1.0, atol=1e-3):
            exp_vec = np.exp(pred_vector - np.max(pred_vector))
            probs = exp_vec / exp_vec.sum()
        else:
            probs = pred_vector
        top_indices = np.argsort(probs)[::-1][:5]
        top5_list = []
        for i in top_indices:
            if i < len(field.category_list):
                cat = field.category_list[i]
                top5_list.append(f"{cat}: {probs[i]:.2f}")
        return ", ".join(top5_list)

    def _digits_to_base_str(self, digits, base):
        # Convert a list or array of digit values to a string representation.
        return "".join(str(int(d)) for d in digits)

    def on_train_batch_end(self, batch, logs=None):
        if not self.samples or (batch + 1) % self.interval_batches != 0:
            return

        print(f"\nBatch {batch + 1}: Reconstruction and Tokenization Demo Results")
        for i, row_dict in enumerate(self.samples):
            print(f"\nSample {i + 1}:")
            table = PrettyTable()
            table.field_names = ["Field", "Original Value", "Reconstructed"]
            table.align = "l"
            for col in ["Original Value", "Reconstructed"]:
                table.max_width[col] = 40

            input_tensors = {}
            valid_field_indices = []
            for field_idx, field in enumerate(self.row_autoencoder.fields):
                try:
                    input_tensors[field.name] = field.transform(row_dict.get(field.name))
                    valid_field_indices.append(field_idx)
                except Exception as e:
                    print(f"Error transforming field '{field.name}' for sample {i+1}: {e}")
                    input_tensors[field.name] = None

            valid_fields = [self.row_autoencoder.fields[idx] for idx in valid_field_indices]
            if not valid_fields:
                print(f"Skipping sample {i+1} due to transformation errors for all fields.")
                continue

            inputs = [np.expand_dims(input_tensors[field.name], axis=0) for field in valid_fields]
            try:
                if not hasattr(self, 'model') or self.model is None:
                    print("Error: model is not available in ReconstructionCallback.")
                    continue
                predictions = self.model.predict(inputs, verbose=0)
                if not isinstance(predictions, list):
                    predictions = [predictions]
                if len(predictions) != len(valid_fields):
                    print(f"Warning: Mismatch between number of predictions ({len(predictions)}) and valid input fields ({len(valid_fields)}) for sample {i+1}.")
                    continue
            except Exception as e:
                print(f"Error during model prediction for sample {i+1}: {e}")
                continue

            prediction_map = {field.name: predictions[idx] for idx, field in enumerate(valid_fields)}

            for field in self.row_autoencoder.fields:
                field_name = field.name
                original_raw = row_dict.get(field_name, "N/A")
                original_str = ", ".join(map(str, original_raw)) if isinstance(original_raw, list) else str(original_raw)

                argmax_str = ""
                reconstructed_str = "N/A"
                top5_str = ""

                if field_name in prediction_map:
                    main_tensor = np.array(prediction_map[field_name][0])
                    if main_tensor.ndim >= 2 and main_tensor.shape[-1] != 1:
                        argmaxed = np.argmax(main_tensor, axis=-1)
                    else:
                        argmaxed = main_tensor
                    argmaxed = argmaxed.flatten()
                    argmax_str = " ".join(str(x) for x in argmaxed[:20])
                    if len(argmaxed) > 20:
                        argmax_str += " ..."

                    try:
                        reconstructed_str = field.to_string(argmaxed)
                    except Exception as e:
                        print(f"Error decoding reconstructed field '{field_name}' for sample {i+1}: {e}")
                        reconstructed_str = "Error decoding"

                    if hasattr(field, "category_list") and field.category_list:
                        # Show top 5 distribution for single-category fields
                        pred_vec = main_tensor.flatten()
                        top5_str = self._top5_predictions(field, pred_vec)
                    elif isinstance(field, NumericDigitCategoryField):
                        # Numeric digit field logic (omitted for brevity)
                        ...
                elif input_tensors[field_name] is None:
                    reconstructed_str = "N/A (transform failed)"

                table.add_row([field_name, original_str, reconstructed_str])

            print(table)


class SequenceReconstructionCallback(tf.keras.callbacks.Callback):
    """
    Callback to display sequence reconstruction examples during training.
    """
    def __init__(
        self,
        sequence_model_instance: 'MoviesToPeopleSequenceAutoencoder',
        db_path: str,
        num_samples: int = 3,
        interval_batches: int = 500
    ):
        super().__init__()
        self.sequence_model = sequence_model_instance
        self.db_path = os.path.abspath(db_path)
        if not os.path.exists(self.db_path):
            logging.error(f"Database path does not exist: {self.db_path}")
        self.num_samples = num_samples
        self.interval_batches = interval_batches
        self.people_sequence_length = sequence_model_instance.people_sequence_length
        self.batch_count = 0

        self.movie_fields: List[Any] = sequence_model_instance.movie_autoencoder_instance.fields
        self.people_fields: List[Any] = sequence_model_instance.people_autoencoder_instance.fields

        if not sequence_model_instance.stats_accumulated:
            logging.warning("SequenceReconstructionCallback created before stats were accumulated.")
            logging.info("Attempting to accumulate stats now...")
            try:
                sequence_model_instance.accumulate_stats()
            except Exception as e:
                logging.error(f"Failed to accumulate stats within callback init: {e}", exc_info=True)
                self.interval_batches = float('inf')

    def _get_sample_data(self) -> List[Dict[str, Any]]:
        samples = []
        try:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                conn.row_factory = sqlite3.Row
                movie_cursor = conn.cursor()
                movie_cursor.execute(f"""
                    SELECT t.tconst
                    FROM titles t
                    INNER JOIN title_genres g ON t.tconst = g.tconst
                    WHERE t.startYear IS NOT NULL
                        AND t.averageRating IS NOT NULL AND t.runtimeMinutes IS NOT NULL
                        AND t.runtimeMinutes >= 5 AND t.startYear >= 1850
                        AND t.titleType IN ('movie', 'tvSeries', 'tvMovie', 'tvMiniSeries')
                        AND t.numVotes >= 10
                    GROUP BY t.tconst
                    ORDER BY RANDOM()
                    LIMIT {self.num_samples * 5}
                """)
                potential_tconsts = [row['tconst'] for row in movie_cursor.fetchall()]

                if not potential_tconsts:
                    logging.warning("Callback Warning: Could not fetch any potential tconsts for sampling.")
                    return []

                sampled_tconsts = random.sample(potential_tconsts, min(self.num_samples, len(potential_tconsts)))

                movie_query = """
                    SELECT
                        t.tconst, t.titleType, t.primaryTitle, t.startYear, t.endYear,
                        t.runtimeMinutes, t.averageRating, t.numVotes,
                        GROUP_CONCAT(g.genre, ',') AS genres
                    FROM titles t
                    LEFT JOIN title_genres g ON t.tconst = g.tconst
                    WHERE t.tconst = ?
                    GROUP BY t.tconst
                """
                people_query = """
                    SELECT
                        p.primaryName, p.birthYear, p.deathYear,
                        GROUP_CONCAT(pp.profession, ',') AS professions
                    FROM people p
                    LEFT JOIN people_professions pp ON p.nconst = pp.nconst
                    INNER JOIN principals pr ON pr.nconst = p.nconst
                    WHERE pr.tconst = ? AND p.birthYear IS NOT NULL
                    GROUP BY p.nconst
                    ORDER BY pr.ordering
                    LIMIT ?
                """

                for tconst in sampled_tconsts:
                    movie_cursor.execute(movie_query, (tconst,))
                    movie_row = movie_cursor.fetchone()
                    if not movie_row: 
                        continue

                    people_cursor = conn.cursor()
                    people_cursor.execute(people_query, (tconst, self.people_sequence_length))
                    people_list = []
                    for people_row in people_cursor.fetchall():
                        person_data = {
                            field.name: people_row[field.name]
                            for field in self.people_fields
                            if field.name in people_row.keys()
                        }
                        if "professions" in person_data:
                            if person_data["professions"]:
                                person_data["professions"] = person_data["professions"].split(',')
                            else:
                                person_data["professions"] = None
                        people_list.append(person_data)

                    if people_list:
                        movie_data = {
                            field.name: movie_row[field.name]
                            for field in self.movie_fields
                            if field.name in movie_row.keys()
                        }
                        if "genres" in movie_data:
                            if movie_data["genres"]:
                                movie_data["genres"] = movie_data["genres"].split(',')
                            else:
                                movie_data["genres"] = []
                        samples.append({**movie_data, "people": people_list})
        except sqlite3.Error as e:
            logging.error(f"Database error in callback _get_sample_data: {e}", exc_info=True)
            return []
        except Exception as e:
            logging.error(f"Unexpected error in callback _get_sample_data: {e}", exc_info=True)
            return []

        return samples

    def _prepare_input_tensors(self, raw_samples: List[Dict[str, Any]]) -> Optional[List[tf.Tensor]]:
        batch_inputs_dict = {field.name: [] for field in self.movie_fields}
        prepared_indices = []

        try:
            for idx, sample in enumerate(raw_samples):
                try:
                    sample_inputs = {}
                    for field in self.movie_fields:
                        transformed_input = field.transform(sample.get(field.name))
                        sample_inputs[field.name] = transformed_input
                    for field_name in batch_inputs_dict.keys():
                        batch_inputs_dict[field_name].append(sample_inputs[field_name])
                    prepared_indices.append(idx)
                except Exception as field_e:
                    logging.warning(f"Callback: Skipping sample '{sample.get('primaryTitle', 'N/A')}' due to error transforming field '{field.name}': {field_e}", exc_info=False)
            if not prepared_indices:
                logging.warning("Callback: No samples could be prepared for input.")
                return None

            batch_input_tensors = []
            for field in self.movie_fields:
                field_batch = tf.stack([batch_inputs_dict[field.name][i] for i, _ in enumerate(prepared_indices)], axis=0)
                batch_input_tensors.append(field_batch)

            return batch_input_tensors, prepared_indices

        except Exception as e:
            logging.error(f"Callback Error preparing batch input tensors: {e}", exc_info=True)
            return None

    def _tensor_to_string(self, field, main_tensor: np.ndarray, flag_tensor: Optional[np.ndarray] = None) -> str:
        try:
            if hasattr(field, "tokenizer") and field.tokenizer is not None:
                if main_tensor.ndim >= 2 and main_tensor.shape[-1] == field.tokenizer.get_vocab_size():
                    main_tensor = np.argmax(main_tensor, axis=-1)
                if main_tensor.ndim > 1:
                    main_tensor = main_tensor.flatten()
            if flag_tensor is not None:
                if flag_tensor.ndim > 1:
                    flag_tensor = flag_tensor.flatten()
                if flag_tensor.size > 1:
                    flag_tensor = flag_tensor[0:1]
                elif flag_tensor.size == 0:
                    flag_tensor = None
            return field.to_string(main_tensor, flag_tensor)
        except Exception as e:
            logging.warning(f"Callback: Error converting tensor to string for field {field.name}: {e}", exc_info=False)
            return "[Conversion Error]"

    def _display_reconstructions(self, raw_samples, predictions_dict, losses: Optional[Dict[str, Any]] = None):
        num_actual_samples = len(raw_samples)
        for i in range(num_actual_samples):
            sample_movie = raw_samples[i]
            movie_table = PrettyTable()
            movie_table.field_names = ["Movie Detail", "Value"]
            for field in self.movie_fields:
                val = sample_movie.get(field.name, "N/A")
                if isinstance(val, list):
                    val = ", ".join(map(str, val))
                else:
                    val = str(val)
                movie_table.add_row([field.name, val])
            print(f"\nMovie Details ({i+1}/{num_actual_samples}):")
            print(movie_table)

            ground_truth_people = sample_movie.get("people", [])
            num_real_people = len(ground_truth_people)
            if num_real_people == 0:
                print("  (No ground truth people found for this movie in the sample data)")
                continue

            table = PrettyTable()
            table.field_names = ["Step", "Field", "Ground Truth", "Prediction"]
            table.align["Field"] = "l"
            table.align["Ground Truth"] = "l"
            table.align["Prediction"] = "l"
            table.max_width["Ground Truth"] = 30
            table.max_width["Prediction"] = 30

            for t in range(num_real_people):
                gt_person_dict = ground_truth_people[t]
                pred_idx = i
                for field in self.people_fields:
                    gt_str = "---"
                    pred_str = "---"

                    try:
                        target_result = field.transform_target(gt_person_dict.get(field.name))
                        if field.optional:
                            gt_main_tensor, gt_flag_tensor = target_result
                            gt_str = self._tensor_to_string(field, gt_main_tensor.numpy(), np.array([0.0]))
                        else:
                            gt_main_tensor = target_result
                            gt_str = self._tensor_to_string(field, gt_main_tensor.numpy(), None)
                    except Exception as e:
                        logging.warning(f"Error processing GT for {field.name} step {t}: {e}", exc_info=False)
                        gt_str = "[GT Error]"

                    pred_main_tensor = None
                    pred_flag_tensor = None
                    main_out_name = f"{field.name}_main_out"
                    flag_out_name = f"{field.name}_flag_out"
                    single_out_name = f"{field.name}_out"
                    try:
                        if isinstance(predictions_dict, dict):
                            if main_out_name in predictions_dict:
                                pred_main_tensor_batch = predictions_dict[main_out_name]
                                if isinstance(pred_main_tensor_batch, np.ndarray) and pred_main_tensor_batch.ndim == 2:
                                    pred_main_tensor_batch = np.expand_dims(pred_main_tensor_batch, axis=0)
                                pred_main_tensor = pred_main_tensor_batch[pred_idx, t]
                                if flag_out_name in predictions_dict:
                                    pred_flag_tensor_batch = predictions_dict[flag_out_name]
                                    if isinstance(pred_flag_tensor_batch, np.ndarray) and pred_flag_tensor_batch.ndim == 2:
                                        pred_flag_tensor_batch = np.expand_dims(pred_flag_tensor_batch, axis=0)
                                    pred_flag_tensor = pred_flag_tensor_batch[pred_idx, t]
                            elif single_out_name in predictions_dict:
                                pred_main_tensor_batch = predictions_dict[single_out_name]
                                if isinstance(pred_main_tensor_batch, np.ndarray) and pred_main_tensor_batch.ndim == 2:
                                    pred_main_tensor_batch = np.expand_dims(pred_main_tensor_batch, axis=0)
                                pred_main_tensor = pred_main_tensor_batch[pred_idx, t]
                                pred_flag_tensor = None
                            else:
                                pred_str = "[Pred Missing]"
                        elif not isinstance(predictions_dict, dict):
                            try:
                                current_pred = predictions_dict
                                if isinstance(current_pred, list) and len(current_pred) == len(self.people_fields):
                                    current_pred = current_pred[self.people_fields.index(field)]
                                if isinstance(current_pred, (np.ndarray, tf.Tensor)):
                                    pred_main_tensor = current_pred[pred_idx, t]
                                else:
                                    pred_str = "[Pred Format Error]"
                            except (IndexError, TypeError):
                                pred_str = "[Pred Access Error]"
                        if pred_str == "---" and pred_main_tensor is not None:
                            pred_str = self._tensor_to_string(field, pred_main_tensor, pred_flag_tensor)
                        elif pred_str == "---":
                            pred_str = "[Pred Not Found]"
                    except IndexError:
                        logging.warning(f"Index out of bounds accessing prediction for sample {pred_idx}, step {t}, field {field.name}.")
                        pred_str = "[Pred Index Error]"
                    except KeyError as e:
                        logging.warning(f"Output key error accessing prediction for {field.name} step {t}: {e}.")
                        pred_str = "[Pred Key Error]"
                    except Exception as e:
                        logging.warning(f"Error processing prediction for {field.name} step {t}: {e}", exc_info=True)
                        pred_str = "[Pred Error]"

                    step_label = f"Person {t+1}"
                    table.add_row([step_label, field.name, gt_str, pred_str])
            print(table)
        print("-" * 80)

    def on_train_batch_end(self, batch, logs=None):
        self.batch_count += 1
        if self.batch_count % self.interval_batches != 0:
            return

        if not self.model:
            logging.warning("Callback: Model reference not available.")
            return

        print(f"\n--- Sequence Reconstruction Example (Batch {self.batch_count}) ---")
        current_lr = self.model.optimizer.learning_rate
        if callable(current_lr):
            try:
                lr_val = current_lr(self.model.optimizer.iterations)
            except TypeError:
                lr_val = tf.keras.backend.get_value(current_lr)
        else:
            lr_val = tf.keras.backend.get_value(current_lr)
        if hasattr(lr_val, 'numpy'):
            lr_val = lr_val.numpy()
        print(f"Current LR: {lr_val:.8f}")

        raw_samples = self._get_sample_data()
        if not raw_samples:
            logging.warning("Callback: No samples found to display.")
            return

        prep_result = self._prepare_input_tensors(raw_samples)
        if prep_result is None:
            logging.error("Callback: Failed to prepare input tensors.")
            return
        input_tensors, prepared_indices = prep_result
        filtered_raw_samples = [raw_samples[i] for i in prepared_indices]
        if not filtered_raw_samples:
            logging.warning("Callback: No raw samples remained after filtering based on preparation.")
            return

        try:
            predictions_dict = self.model.predict_on_batch(input_tensors)
            if isinstance(predictions_dict, dict):
                predictions_dict = {k: v.numpy() if tf.is_tensor(v) else v for k, v in predictions_dict.items()}
            elif tf.is_tensor(predictions_dict):
                predictions_dict = predictions_dict.numpy()
            elif isinstance(predictions_dict, list):
                predictions_dict = [v.numpy() if tf.is_tensor(v) else v for v in predictions_dict]
        except Exception as e:
            logging.error(f"Error during model prediction_on_batch for callback: {e}", exc_info=True)
            return

        self._display_reconstructions(filtered_raw_samples, predictions_dict, losses=logs)
