"""CNN training loop with early stopping on validation macro-F1."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from .models import SmallCNN


@torch.inference_mode()
def predict(model: nn.Module, X: np.ndarray, batch_size: int = 512) -> tuple[np.ndarray, float]:
    """Return (predicted labels, logits) in eval mode."""
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        out.append(model(torch.from_numpy(X[i:i + batch_size])))
    logits = torch.cat(out)
    return logits.argmax(1).numpy(), logits


def train_cnn(X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
              cfg: dict, seed: int, log_every: int = 0) -> tuple[SmallCNN, list[dict], dict]:
    """Train a SmallCNN; return the best-on-validation model, history and summary.

    The test set is never passed to this function.
    """
    m = cfg["cnn"]
    t = cfg["training"]
    model = SmallCNN(in_channels=X_tr.shape[1], widths=tuple(m["widths"]),
                     kernels=tuple(m["kernels"]), dropout=m["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)

    Xt, yt = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
    yv = torch.from_numpy(y_val)
    best_f1, best_epoch, best_state = -1.0, -1, None
    history = []
    start = time.perf_counter()

    for epoch in range(1, t["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        total, n = 0.0, 0
        for i in range(0, len(perm), t["batch_size"]):
            idx = perm[i:i + t["batch_size"]]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            n += len(idx)

        pred, logits = predict(model, X_val)
        val_loss = loss_fn(logits, yv).item()
        val_f1 = f1_score(y_val, pred, average="macro")
        val_acc = float((pred == y_val).mean())
        history.append({"epoch": epoch, "train_loss": total / n, "val_loss": val_loss,
                        "val_acc": val_acc, "val_macro_f1": val_f1})
        if log_every and epoch % log_every == 0:
            print(f"    epoch {epoch:3d} train_loss={total / n:.4f} "
                  f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= t["patience"]:
            break

    model.load_state_dict(best_state)
    model.eval()
    summary = {"best_epoch": best_epoch, "epochs_run": len(history),
               "best_val_macro_f1": best_f1,
               "train_seconds": time.perf_counter() - start}
    return model, history, summary
