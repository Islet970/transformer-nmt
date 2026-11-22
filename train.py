from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from config import Config, apply_overrides, resolve_sp_model, validate_data_dir
from data.dataset import build_dataloaders
from data.tokenizer import load_tokenizer
from evaluate import evaluate_loader
from models.factory import build_model
from utils import (
    AverageMeter,
    Logger,
    Timer,
    causal_mask,
    count_parameters,
    format_seconds,
    get_device,
    make_pad_mask,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/transformer.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--sp-model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--beam", type=int, default=None)
    parser.add_argument("--val-max-batches", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def build_optimizer(model: nn.Module, cfg: Config):
    if cfg.model == "transformer":
        optimizer = Adam(
            model.parameters(),
            lr=cfg.lr,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=cfg.weight_decay,
        )
        if cfg.warmup > 0:
            def lr_lambda(step: int) -> float:
                step = max(1, step)
                return (cfg.d_model ** -0.5) * min(
                    step ** -0.5, step * cfg.warmup ** -1.5
                )
            scheduler = LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = None
        return optimizer, scheduler

    optimizer = Adam(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=cfg.weight_decay,
    )
    return optimizer, None


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    scaler,
    device,
    cfg: Config,
    epoch: int,
    global_step: int,
    logger: Logger,
    use_amp: bool,
    max_steps: int = 0,
):
    model.train()
    loss_meter = AverageMeter()
    timer = Timer()

    for step, (src, tgt) in enumerate(loader):
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask = make_pad_mask(src).to(device)
        tgt_mask = (
            make_pad_mask(tgt_in).to(device)
            & causal_mask(tgt_in.size(1), device)
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1),
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        loss_meter.update(loss.item(), src.size(0))
        global_step += 1

        if step % cfg.log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.log(
                f"epoch {epoch} step {step}/{len(loader)} "
                f"loss {loss.item():.4f} lr {lr:.6f} "
                f"elapsed {format_seconds(timer.elapsed)}"
            )

        if max_steps and global_step >= max_steps:
            logger.log(f"reached max_steps={max_steps}, stop epoch early")
            break

    return loss_meter.avg, global_step, timer.elapsed


def save_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    scaler,
    cfg: Config,
    epoch: int,
    global_step: int,
    best_bleu: float,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_bleu": best_bleu,
            "config": cfg.to_dict(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    cfg = apply_overrides(
        cfg,
        argparse.Namespace(
            data_dir=args.data_dir,
            sp_model=args.sp_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            limit_train=args.limit_train,
            limit_val=args.limit_val,
            run_name=args.run_name,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            beam=args.beam,
            val_max_batches=args.val_max_batches,
            amp=False if args.no_amp else None,
        ),
    )

    validate_data_dir(cfg.data_dir)
    sp_model = resolve_sp_model(cfg, args.sp_model)
    tokenizer = load_tokenizer(sp_model)

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    use_amp = bool(cfg.amp and device.type == "cuda")

    run_name = cfg.run_name or f"{cfg.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(cfg.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger = Logger(os.path.join(run_dir, "train.log"))
    cfg.save_json(os.path.join(run_dir, "config.json"))

    logger.log(f"run_dir: {run_dir}")
    logger.log(f"device: {device} | amp: {use_amp}")
    logger.log(f"data_dir: {cfg.data_dir}")
    logger.log(f"sp_model: {sp_model}")

    train_loader, val_loader, _ = build_dataloaders(cfg, tokenizer)
    logger.log(
        f"train batches: {len(train_loader)} | "
        f"valid batches: {len(val_loader)} | "
        f"vocab: {tokenizer.vocab_size}"
    )

    model = build_model(cfg, tokenizer.vocab_size).to(device)
    logger.log(f"model: {cfg.model} | params: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id,
        label_smoothing=cfg.label_smoothing,
    )
    optimizer, scheduler = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    global_step = 0
    best_bleu = 0.0

    if args.resume:
        last_path = os.path.join(run_dir, "last.pt")
        if not os.path.exists(last_path):
            raise FileNotFoundError(f"--resume set but {last_path} does not exist")
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        best_bleu = ckpt.get("best_bleu", 0.0)
        logger.log(
            f"resumed from epoch {start_epoch}, global_step {global_step}, "
            f"best_bleu {best_bleu:.2f}"
        )

    loss_csv_path = os.path.join(run_dir, "loss.csv")
    write_header = not os.path.exists(loss_csv_path)
    loss_csv = open(loss_csv_path, "a", newline="", encoding="utf-8")
    loss_writer = csv.writer(loss_csv)
    if write_header:
        loss_writer.writerow(["epoch", "train_loss", "val_bleu", "val_chrf", "seconds"])
        loss_csv.flush()

    no_improve = 0
    epochs_run = 0
    for epoch in range(start_epoch, cfg.epochs):
        logger.log(f"===== epoch {epoch}/{cfg.epochs - 1} =====")
        train_loss, global_step, seconds = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            scaler,
            device,
            cfg,
            epoch,
            global_step,
            logger,
            use_amp,
            max_steps=cfg.max_steps,
        )
        epochs_run = epoch - start_epoch + 1

        val_result = evaluate_loader(
            model,
            val_loader,
            tokenizer,
            device,
            max_len=cfg.max_len,
            beam=cfg.beam,
            length_penalty=cfg.length_penalty,
            max_batches=cfg.val_max_batches,
        )
        val_bleu = val_result["bleu"]
        val_chrf = val_result["chrf"]

        logger.log(
            f"epoch {epoch} done | loss {train_loss:.4f} | "
            f"BLEU {val_bleu:.2f} | chrF {val_chrf:.2f} | "
            f"{format_seconds(seconds)}"
        )

        loss_writer.writerow([epoch, f"{train_loss:.6f}", val_bleu, val_chrf, f"{seconds:.1f}"])
        loss_csv.flush()

        save_checkpoint(
            os.path.join(run_dir, "last.pt"),
            model,
            optimizer,
            scheduler,
            scaler,
            cfg,
            epoch,
            global_step,
            best_bleu,
        )

        if val_bleu > best_bleu:
            best_bleu = val_bleu
            no_improve = 0
            save_checkpoint(
                os.path.join(run_dir, "best.pt"),
                model,
                optimizer,
                scheduler,
                scaler,
                cfg,
                epoch,
                global_step,
                best_bleu,
            )
            logger.log(f"new best BLEU: {best_bleu:.2f}")
        else:
            no_improve += 1
            if cfg.patience > 0:
                logger.log(f"no improvement ({no_improve}/{cfg.patience})")

        save_json(
            os.path.join(run_dir, "val_last.json"),
            {k: v for k, v in val_result.items() if k != "predictions"},
        )

        if cfg.patience > 0 and no_improve >= cfg.patience:
            logger.log(f"early stopping at epoch {epoch}")
            break

        if cfg.max_steps and global_step >= cfg.max_steps:
            logger.log("max_steps reached, stop training")
            break

    loss_csv.close()
    save_json(
        os.path.join(run_dir, "summary.json"),
        {
            "run_name": run_name,
            "best_bleu": best_bleu,
            "epochs_run": epochs_run,
            "global_step": global_step,
            "params": count_parameters(model),
            "config": cfg.to_dict(),
        },
    )
    logger.log(f"done. best BLEU: {best_bleu:.2f}")
    logger.close()


if __name__ == "__main__":
    main()
