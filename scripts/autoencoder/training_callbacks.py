from itertools import islice
import datetime
import logging
from pathlib import Path
import random
import sqlite3
import numpy as np
from prettytable import PrettyTable, TableStyle
from typing import Any, List, Dict, Optional
import tensorflow as tf
from .fields import BaseField, SPECIAL_PAD, SPECIAL_START, SPECIAL_END, NumericDigitCategoryField, TextField  # Import special tokens here
import os

def _sample_random_person(conn, tconst):
    q = """
        SELECT p.primaryName,
               p.birthYear,
               p.deathYear,
               GROUP_CONCAT(pp.profession, ',')
        FROM   people p
        LEFT   JOIN people_professions pp ON pp.nconst = p.nconst
        INNER  JOIN principals pr         ON pr.nconst = p.nconst
        WHERE pr.tconst = ? 
          AND p.birthYear IS NOT NULL
        GROUP  BY p.nconst
        HAVING COUNT(pp.profession) > 0
        ORDER  BY RANDOM()
        LIMIT  1
    """
    r = conn.execute(q, (tconst,)).fetchone()
    if not r:
        return None
    return {
        "primaryName": r[0],
        "birthYear":   r[1],
        "deathYear":   r[2],
        "professions": r[3].split(',') if r[3] else None,
    }


def _norm(x): return np.linalg.norm(x) + 1e-9
def _cos(a, b): return float(np.dot(a, b) / (_norm(a) * _norm(b)))

class JointReconstructionCallback(tf.keras.callbacks.Callback):
    def __init__(self, movie_ae, person_ae, db_path, interval_batches=200, num_samples=4, table_width=38):
        super().__init__()
        self.m_ae = movie_ae
        self.p_ae = person_ae
        self.every = interval_batches
        self.w = table_width
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        movies = list(islice(movie_ae.row_generator(), 50000))
        self.pairs = []
        for m in random.sample(movies, min(num_samples * 3, len(movies))):
            p = self._sample_person(m["tconst"])
            if p:
                self.pairs.append((m, p))
            if len(self.pairs) == num_samples:
                break
        neg_movies = random.sample(movies, min(256, len(movies)))
        self.neg_people_latents = []
        for m in neg_movies:
            p = self._sample_person(m["tconst"])
            if p:
                z = self._encode(self.p_ae, p)
                self.neg_people_latents.append(z)
        self.neg_people_latents = np.stack(self.neg_people_latents)

    def _sample_person(self, tconst):
        q = """
        SELECT p.primaryName, p.birthYear, p.deathYear, GROUP_CONCAT(pp.profession, ',')
        FROM people p
        LEFT JOIN people_professions pp ON pp.nconst = p.nconst
        INNER JOIN principals pr ON pr.nconst = p.nconst
        WHERE pr.tconst = ? AND p.birthYear IS NOT NULL
        GROUP BY p.nconst
        HAVING COUNT(pp.profession) > 0
        ORDER BY RANDOM()
        LIMIT 1
        """
        r = self.conn.execute(q, (tconst,)).fetchone()
        if not r:
            return None
        return {
            "primaryName": r[0],
            "birthYear": r[1],
            "deathYear": r[2],
            "professions": r[3].split(',') if r[3] else None,
        }

    def _encode(self, ae, row):
        xs = [tf.expand_dims(f.transform(row.get(f.name)), 0) for f in ae.fields]
        return ae.encoder(xs, training=False).numpy()[0]

    def _recon(self, ae, z):
        return ae.reconstruct_row(z)

    def _cos(self, a, b):
        a_norm = a / (np.linalg.norm(a) + 1e-9)
        b_norm = b / (np.linalg.norm(b) + 1e-9)
        return float(np.dot(a_norm, b_norm))

    def _rank(self, z_m, z_p):
        sims = np.dot(self.neg_people_latents, z_m) / (
            (np.linalg.norm(self.neg_people_latents, axis=1) + 1e-9) * (np.linalg.norm(z_m) + 1e-9)
        )
        sims = np.append(sims, self._cos(z_m, z_p))
        return int((-sims).argsort().tolist().index(len(sims) - 1)) + 1

    def _tensor_to_string(self, field, main_tensor, flag_tensor=None):
        try:
            if isinstance(field, NumericDigitCategoryField):
                arr = np.array(main_tensor)
                if arr.ndim == 3:
                    return field.to_string(arr)
                arr = arr.flatten()
                return field.to_string(arr)
            if hasattr(field, "tokenizer") and field.tokenizer is not None:
                arr = np.array(main_tensor)
                if arr.ndim >= 2 and arr.shape[-1] == field.tokenizer.get_vocab_size():
                    arr = np.argmax(arr, axis=-1)
            else:
                arr = np.array(main_tensor).flatten()
            if flag_tensor is not None:
                ft = np.array(flag_tensor)
                if ft.ndim > 1:
                    ft = ft.flatten()
                if ft.size > 1:
                    ft = ft[:1]
                elif ft.size == 0:
                    ft = None
            else:
                ft = None
            return field.to_string(arr, ft)
        except Exception:
            return "[Conversion Error]"

    def on_train_batch_end(self, batch, logs=None):
        if (batch + 1) % self.every:
            return
        for m_row, p_row in self.pairs:
            z_m = self._encode(self.m_ae, m_row)
            z_p = self._encode(self.p_ae, p_row)
            cos = self._cos(z_m, z_p)
            l2 = float(np.linalg.norm(z_m - z_p))
            ang = float(np.degrees(np.arccos(max(min(cos, 1.0), -1.0))))
            rank = self._rank(z_m, z_p)
            m_rec = self._recon(self.m_ae, z_m)
            p_rec = self._recon(self.p_ae, z_p)
            tm = PrettyTable(["Movie field", "orig", "recon"])
            for f in self.m_ae.fields:
                o = m_row.get(f.name, "")
                r_raw = m_rec.get(f.name, "")
                if isinstance(r_raw, np.ndarray):
                    r = self._tensor_to_string(f, r_raw)
                else:
                    r = r_raw
                tm.add_row([f.name, str(o)[: self.w], str(r)[: self.w]])
            tp = PrettyTable(["Person field", "orig", "recon"])
            for f in self.p_ae.fields:
                o = p_row.get(f.name, "")
                r_raw = p_rec.get(f.name, "")
                if isinstance(r_raw, np.ndarray):
                    r = self._tensor_to_string(f, r_raw)
                else:
                    r = r_raw
                tp.add_row([f.name, str(o)[: self.w], str(r)[: self.w]])
            bar_len = 20
            bar = "#" * int(cos * bar_len) + "-" * (bar_len - int(cos * bar_len))
            print(
                f"\n--- joint recon ---\n"
                f"cos={cos:.3f}  angle={ang:5.1f}°  l2={l2:.3f}  rank@neg={rank}\n"
                f"[{bar}]\n"
                f"{tm}\n{tp}"
            )


class TensorBoardPerBatchLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir: Path, log_interval: int = 1):
        super().__init__()
        self.log_interval = log_interval
        now = datetime.datetime.now()
        run_id = f"{now:%Y%m%d-%H%M%S}"
        self.file_writer = tf.summary.create_file_writer(str(log_dir / run_id))

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}

        # ▲ read the variable’s *value*, not the variable itself
        step = int(self.model.optimizer.iterations.numpy())

        total_loss = float(logs.get("loss", 0.0))

        lr = self.model.optimizer.learning_rate
        lr_value = (
            float(lr(step)) if callable(lr)          # learning‑rate schedule
            else float(tf.keras.backend.get_value(lr))
        )

        with self.file_writer.as_default():
            tf.summary.scalar("loss/total",      total_loss,  step=step)
            tf.summary.scalar("learning_rate",   lr_value,    step=step)
            for k, v in logs.items():
                if k.endswith("_loss") and k != "loss":
                    tf.summary.scalar(f"loss/{k[:-5]}", float(v), step=step)
            self.file_writer.flush()




class ModelSaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, row_autoencoder, output_dir):
        super().__init__()
        self.row_autoencoder = row_autoencoder
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        self.row_autoencoder.save_model()
        print(f"Model saved to {self.output_dir} at the end of epoch {epoch + 1}")


class ReconstructionCallback(tf.keras.callbacks.Callback):
    def __init__(self, interval_batches, row_autoencoder, db_path, num_samples=5):
        super().__init__()
        self.interval_batches = interval_batches
        self.row_autoencoder = row_autoencoder
        self.db_path = db_path
        self.num_samples = num_samples

        if not self.row_autoencoder.stats_accumulated:
            self.row_autoencoder.accumulate_stats()

        all_rows = list(row_autoencoder.row_generator())
        actual_num_samples = min(num_samples, len(all_rows))
        if actual_num_samples > 0:
            self.samples = random.sample(all_rows, actual_num_samples)
        else:
            self.samples = []

    def _tensor_to_string(self, field, main_tensor: np.ndarray, flag_tensor: np.ndarray = None) -> str:
        try:
            if isinstance(field, NumericDigitCategoryField):
                if main_tensor.ndim == 3:
                    return field.to_string(main_tensor)
                arr = np.array(main_tensor)
                return field.to_string(arr)
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
        except Exception:
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
                top5_list.append(f"{cat}:{probs[i]:.2f}")
        return ", ".join(top5_list)

    def _digits_to_base_str(self, digits, base):
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
                except Exception:
                    input_tensors[field.name] = None

            valid_fields = [self.row_autoencoder.fields[idx] for idx in valid_field_indices]
            if not valid_fields:
                print(f"Skipping sample {i+1} due to transformation errors for all fields.")
                continue

            inputs = [np.expand_dims(input_tensors[field.name], axis=0) for field in valid_fields]
            try:
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

                reconstructed_str = "N/A"
                if field_name in prediction_map:
                    main_tensor = np.array(prediction_map[field_name][0])
                    flag_tensor = None
                    if field.optional:
                        flag_tensor = np.array(prediction_map.get(f"{field.name}_flag_out", [np.array([])])[0])
                    reconstructed_str = self._tensor_to_string(field, main_tensor, flag_tensor)
                    if hasattr(field, "category_list") and field.category_list and not isinstance(field, NumericDigitCategoryField):
                        top5_str = self._top5_predictions(field, main_tensor.flatten())
                        reconstructed_str += f" ({top5_str})"
                elif input_tensors[field_name] is None:
                    reconstructed_str = "N/A (transform failed)"

                table.add_row([field_name, original_str, reconstructed_str])

            print(table)



class SequenceReconstructionCallback(tf.keras.callbacks.Callback):
    """
    Displays sequence reconstruction examples during training of MoviesToPeopleSequenceAutoencoder.
    Samples a fixed set of movie→people sequences at init, then every `interval_batches` it runs
    a single predict_on_batch and prints ground truth vs. reconstruction for each person.
    """
    def __init__(
        self,
        sequence_model_instance: 'MoviesToPeopleSequenceAutoencoder',
        num_samples: int = 3,
        interval_batches: int = 500
    ):
        super().__init__()
        self.seq_model = sequence_model_instance
        self.interval = interval_batches

        # sample a few full rows
        all_rows = list(islice(self.seq_model.row_generator(), 50000))
        self.samples = random.sample(all_rows, min(num_samples, len(all_rows)))

        self.movie_fields = self.seq_model.movie_autoencoder_instance.fields
        self.people_fields = self.seq_model.people_autoencoder_instance.fields
        self.seq_len = self.seq_model.people_sequence_length
        self.batch_counter = 0

    def _tensor_to_string(self, field, tensor: np.ndarray, flag: np.ndarray = None) -> str:
        """
        Convert raw model output for a single field at one time‐step into a string.
        Special‐case numeric‐digit fields by argmaxing over the base dimension.
        """
        try:
            arr = np.array(tensor)
            # NumericDigitCategoryField outputs shape (positions, base)
            if isinstance(field, NumericDigitCategoryField):
                if arr.ndim == 2:
                    # pick the most likely digit at each position
                    arr = np.argmax(arr, axis=-1)
                # now arr should be shape (positions,)
                return field.to_string(arr)
            
            # TextField or one‐hot‐style: argmax over last dim if it matches vocab
            if hasattr(field, "tokenizer") and field.tokenizer:
                if arr.ndim >= 2 and arr.shape[-1] == field.tokenizer.get_vocab_size():
                    arr = np.argmax(arr, axis=-1)
            
            # flatten anything left
            if arr.ndim > 1:
                arr = arr.flatten()
            
            return field.to_string(arr, flag)
        except Exception as e:
            logging.warning(f"Tensor→string error for {field.name}: {e}")
            return "[Conversion Error]"

    def on_train_batch_end(self, batch, logs=None):
        self.batch_counter += 1
        if self.batch_counter % self.interval != 0:
            return

        # fetch LR value
        lr_tensor = self.model.optimizer.learning_rate
        try:
            lr_val = tf.keras.backend.get_value(lr_tensor)
        except Exception:
            lr_val = lr_tensor.numpy() if hasattr(lr_tensor, "numpy") else float(lr_tensor)
        print(f"\n--- Sequence Reconstruction (batch {self.batch_counter}) LR={lr_val:.2e} ---")

        # prepare one batch of movie inputs
        batch_inputs = []
        for f in self.movie_fields:
            vals = [f.transform(row[f.name]) for row in self.samples]
            batch_inputs.append(tf.stack(vals, axis=0))

        # run predict_on_batch
        try:
            raw_preds = self.model.predict_on_batch(batch_inputs)
        except Exception as e:
            logging.error(f"Predict‐on‐batch failed: {e}", exc_info=True)
            return

        # map output names → numpy arrays
        preds = {}
        for name, arr in zip(self.model.output_names, raw_preds):
            if tf.is_tensor(arr):
                arr = arr.numpy()
            preds[name] = np.array(arr)

        # display each sample
        for i, row in enumerate(self.samples):
            # movie info
            movie_tbl = PrettyTable(["Movie Field", "Value"])
            for f in self.movie_fields:
                v = row.get(f.name, "")
                movie_tbl.add_row([f.name, ", ".join(v) if isinstance(v, list) else str(v)])
            print(f"\nSample {i+1} Movie:")
            print(movie_tbl)

            # people reconstruction
            people_tbl = PrettyTable(["Person #", "Field", "Ground Truth", "Reconstruction"])
            for t in range(self.seq_len):
                person = row["people"][t] if t < len(row["people"]) else {}
                for f in self.people_fields:
                    # ground truth
                    gt = person.get(f.name, None)
                    gt_str = ", ".join(gt) if isinstance(gt, list) else str(gt)

                    # choose output keys
                    main_key = f"{f.name}_main_out" if f.optional else f"{f.name}_out"
                    flag_key = f"{f.name}_flag_out" if f.optional else None

                    pm = preds.get(main_key)
                    pf = preds.get(flag_key) if flag_key else None

                    if pm is not None:
                        pred_tensor = pm[i, t]
                        flag_tensor = pf[i, t] if pf is not None else None
                        rec_str = self._tensor_to_string(f, pred_tensor, flag_tensor)
                    else:
                        rec_str = "[no pred]"

                    people_tbl.add_row([t+1, f.name, gt_str, rec_str])

            print(people_tbl)

        print("-" * 80)
