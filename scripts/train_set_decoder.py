import argparse
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from config import project_config, ensure_dirs
from scripts.set_decoder.training import (
    build_set_decoder_trainer,
    _compute_cost_matrices,
    _hungarian,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Train set decoder: movie latent -> set of people latents"
    )
    args = parser.parse_args()

    cfg = project_config
    ensure_dirs(cfg)

    db_path = cfg.db_path

    (
        model,
        opt,
        loader,
        mov_ae,
        per_ae,
        run_logger,
        recon_logger,
        loss_cfg,
    ) = build_set_decoder_trainer(cfg, db_path=db_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = int(getattr(cfg, "set_decoder_epochs", 3))
    save_interval = int(getattr(cfg, "set_decoder_save_interval", 1000))
    flush_interval = int(getattr(cfg, "flush_interval", 250))

    w_latent = loss_cfg["w_latent"]
    w_recon = loss_cfg["w_recon"]
    w_presence = loss_cfg["w_presence"]
    w_null = loss_cfg["w_null"]

    num_slots = int(getattr(cfg, "set_decoder_slots", 10))

    global_step = 0
    best_loss = None

    for epoch in range(num_epochs):
        pbar = tqdm(loader, dynamic_ncols=True)
        pbar.set_description(f"set-decoder epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            z_movies, Z_gt, mask, Y_gt_fields = batch

            z_movies = z_movies.to(device, non_blocking=True)
            Z_gt = Z_gt.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            Y_gt_fields = [y.to(device, non_blocking=True) for y in Y_gt_fields]

            model.train()
            opt.zero_grad()

            z_slots, presence_logits = model(z_movies)

            C_match_list, C_lat_list, C_rec_list = _compute_cost_matrices(
                per_ae,
                z_slots,
                Z_gt,
                Y_gt_fields,
                mask,
                w_latent=w_latent,
                w_recon=w_recon,
            )

            B = z_slots.shape[0]
            N = z_slots.shape[1]

            total_latent_loss = torch.zeros((), device=device)
            total_recon_loss = torch.zeros((), device=device)
            total_presence_loss = torch.zeros((), device=device)
            total_null_loss = torch.zeros((), device=device)

            count_matched_batches = 0
            count_presence_batches = 0
            count_null_slots = 0

            total_gt = 0
            total_pred_on = 0

            recalls = []
            precisions = []
            card_errors = []

            for b in range(B):
                k_b = int(mask[b].sum().item())
                logits_b = presence_logits[b]

                if k_b == 0 or C_match_list[b] is None:
                    target_pres = torch.zeros_like(logits_b)
                    loss_pres_b = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits_b,
                        target_pres,
                        reduction="mean",
                    )
                    total_presence_loss = total_presence_loss + loss_pres_b
                    count_presence_batches += 1

                    probs_b = torch.sigmoid(logits_b)
                    pred_on = int((probs_b > 0.5).sum().item())
                    total_pred_on += pred_on
                    total_gt += 0
                    recalls.append(1.0 if pred_on == 0 else 0.0)
                    precisions.append(1.0 if pred_on == 0 else 0.0)
                    card_errors.append(abs(pred_on - 0))
                    continue

                C_match = C_match_list[b]
                C_lat = C_lat_list[b]
                C_rec = C_rec_list[b]

                rows, cols = _hungarian(C_match)

                if rows.numel() == 0:
                    target_pres = torch.zeros_like(logits_b)
                    loss_pres_b = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits_b,
                        target_pres,
                        reduction="mean",
                    )
                    total_presence_loss = total_presence_loss + loss_pres_b
                    count_presence_batches += 1

                    probs_b = torch.sigmoid(logits_b)
                    pred_on = int((probs_b > 0.5).sum().item())
                    total_pred_on += pred_on
                    total_gt += k_b
                    recalls.append(0.0)
                    precisions.append(0.0 if pred_on > 0 else 1.0)
                    card_errors.append(abs(pred_on - k_b))
                    continue

                idx_rows = rows.to(device)
                idx_cols = cols.to(device)

                valid = idx_cols < k_b
                idx_rows = idx_rows[valid]
                idx_cols = idx_cols[valid]

                if idx_rows.numel() == 0:
                    target_pres = torch.zeros_like(logits_b)
                    loss_pres_b = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits_b,
                        target_pres,
                        reduction="mean",
                    )
                    total_presence_loss = total_presence_loss + loss_pres_b
                    count_presence_batches += 1

                    probs_b = torch.sigmoid(logits_b)
                    pred_on = int((probs_b > 0.5).sum().item())
                    total_pred_on += pred_on
                    total_gt += k_b
                    recalls.append(0.0)
                    precisions.append(0.0 if pred_on > 0 else 1.0)
                    card_errors.append(abs(pred_on - k_b))
                    continue

                matched_lat = C_lat[idx_rows, idx_cols].mean()
                matched_rec = C_rec[idx_rows, idx_cols].mean()

                total_latent_loss = total_latent_loss + matched_lat
                total_recon_loss = total_recon_loss + matched_rec
                count_matched_batches += 1

                target_pres = torch.zeros_like(logits_b)
                target_pres[idx_rows] = 1.0
                loss_pres_b = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits_b,
                    target_pres,
                    reduction="mean",
                )
                total_presence_loss = total_presence_loss + loss_pres_b
                count_presence_batches += 1

                probs_b = torch.sigmoid(logits_b)
                pred_on = int((probs_b > 0.5).sum().item())
                total_pred_on += pred_on
                total_gt += k_b

                mask_null = torch.ones(N, dtype=torch.bool, device=device)
                mask_null[idx_rows] = False
                if mask_null.any():
                    z_null = z_slots[b, mask_null, :]
                    total_null_loss = total_null_loss + z_null.pow(2).mean()
                    count_null_slots += 1

                matched_b = int(idx_rows.numel())
                recall_b = matched_b / max(k_b, 1)
                prec_b = matched_b / max(pred_on, 1)
                recalls.append(recall_b)
                precisions.append(prec_b)
                card_errors.append(abs(pred_on - k_b))

            if count_matched_batches > 0:
                latent_loss = total_latent_loss / float(count_matched_batches)
                recon_loss = total_recon_loss / float(count_matched_batches)
            else:
                latent_loss = torch.zeros((), device=device)
                recon_loss = torch.zeros((), device=device)

            if count_presence_batches > 0:
                presence_loss = total_presence_loss / float(count_presence_batches)
            else:
                presence_loss = torch.zeros((), device=device)

            if count_null_slots > 0:
                null_loss = total_null_loss / float(count_null_slots)
            else:
                null_loss = torch.zeros((), device=device)

            total_loss = (
                w_latent * latent_loss
                + w_recon * recon_loss
                + w_presence * presence_loss
                + w_null * null_loss
            )

            total_loss.backward()
            opt.step()

            mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
            mean_prec = sum(precisions) / len(precisions) if precisions else 0.0
            mean_card_err = sum(card_errors) / len(card_errors) if card_errors else 0.0

            if run_logger:
                run_logger.add_scalar("set_decoder/loss_total", float(total_loss.detach().cpu()), global_step)
                run_logger.add_scalar("set_decoder/loss_latent", float(latent_loss.detach().cpu()), global_step)
                run_logger.add_scalar("set_decoder/loss_recon", float(recon_loss.detach().cpu()), global_step)
                run_logger.add_scalar("set_decoder/loss_presence", float(presence_loss.detach().cpu()), global_step)
                run_logger.add_scalar("set_decoder/loss_null", float(null_loss.detach().cpu()), global_step)
                run_logger.add_scalar("set_decoder/recall", float(mean_recall), global_step)
                run_logger.add_scalar("set_decoder/precision", float(mean_prec), global_step)
                run_logger.add_scalar("set_decoder/cardinality_error", float(mean_card_err), global_step)

            pbar.set_postfix(
                loss=f"{float(total_loss.detach().cpu()):.4f}",
                rec=f"{float(mean_recall):.3f}",
                prec=f"{float(mean_prec):.3f}",
                card_err=f"{float(mean_card_err):.3f}",
            )

            if (global_step + 1) % flush_interval == 0 and run_logger:
                run_logger.flush()

            if (global_step + 1) % save_interval == 0:
                out_dir = Path(cfg.model_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / "SetDecoder.pt"
                torch.save(model.state_dict(), path)
                logging.info(f"saved checkpoint to {path}")

            if hasattr(loader.dataset, "movies"):
                sample_tconsts = loader.dataset.movies[: z_movies.size(0)]
            else:
                sample_tconsts = [""] * z_movies.size(0)

            if recon_logger:
                recon_logger.step(
                    global_step=global_step,
                    sample_tconsts=sample_tconsts,
                    z_movies=z_movies.detach().cpu(),
                    mask=mask.detach().cpu(),
                    run_logger=run_logger,
                )

            if best_loss is None or float(total_loss.detach().cpu()) < best_loss:
                best_loss = float(total_loss.detach().cpu())

            global_step += 1

    out_dir = Path(cfg.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "SetDecoder_final.pt"
    torch.save(model.state_dict(), final_path)
    logging.info(f"saved final model to {final_path}")

    if run_logger:
        run_logger.close()


if __name__ == "__main__":
    main()
