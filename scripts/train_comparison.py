"""Train small Graph-KAN and Baseline GNN models for comparison."""
from collections.abc import Sequence
from typing import Optional
import argparse

import json

import torch
from torch_geometric.loader import DataLoader
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from nbody_gkan.data.dataset import NBodyDataset, normalize_feature_spec
from nbody_gkan.models import OrdinaryGraphKAN, OGN
from nbody_gkan.device import get_device
from nbody_gkan.models.model_loader import ModelLoader

from nbody_gkan.training import GNNTrainer, KANTrainer

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Train Graph-KAN and Baseline GNN models for comparison.")
    # Data
    parser.add_argument("--train_data", type=str, default="data/train.npz")
    parser.add_argument("--val_data", type=str, default="data/val.npz")
    parser.add_argument(
        "--include_velocity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include velocity components in node features (default: true).",
    )
    parser.add_argument(
        "--input_features",
        type=str,
        default=None,
        help=(
            "Optional JSON feature spec, e.g. "
            "'{\"include\": [\"pos\", \"mass\"]}'."
        ),
    )
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/comparison")
    # GNN hyperparameters
    parser.add_argument("--hidden", type=int, default=200)
    parser.add_argument("--msg_dim", type=int, default=32)
    # GKAN hyperparameters
    parser.add_argument("--kan_msg_width", type=str, default=None,
                        help="pykan width list for message net, e.g. '[12,[8,1],6]'")
    parser.add_argument("--kan_node_width", type=str, default=None,
                        help="pykan width list for node net, e.g. '[10,8,3]'")
    parser.add_argument("--kan_msg_dim", type=int, default=None,
                        help="Optional override; inferred from kan_msg_width when omitted.")
    parser.add_argument("--kan_msg_mult_arity", type=str, default="2",
                        help="Multiplication arity (int) or JSON list-of-lists per layer, e.g. '2' or '[[],[2,3],[]]'.")
    parser.add_argument("--kan_node_mult_arity", type=str, default="2",
                        help="Multiplication arity (int) or JSON list-of-lists per layer, e.g. '2' or '[[],[2,3],[]]'.")
    parser.add_argument("--kan_grid_size", type=int, default=5)
    parser.add_argument("--kan_base_fun", type=str, default=None,
                        help="Base function for KAN activations (e.g., 'identity' for linear start).")
    parser.add_argument("--kan_noise_scale", type=float, default=None,
                        help="Spline noise scale; set 0 with base_fun=identity for linear init.")
    parser.add_argument("--kan_scale_base_mu", type=float, default=None,
                        help="Mean for base function scale; set 1 for deterministic identity.")
    parser.add_argument("--kan_scale_base_sigma", type=float, default=None,
                        help="Std for base function scale; set 0 for deterministic identity.")
    parser.add_argument("--kan_lbfgs_lr", type=float, default=1.0)
    parser.add_argument("--kan_lbfgs_max_iter", type=int, default=10)
    parser.add_argument("--kan_lbfgs_max_eval", type=int, default=None)
    parser.add_argument("--kan_lbfgs_line_search_fn", type=str, default="strong_wolfe",
                        help="LBFGS line search: 'strong_wolfe' or 'none' for faster steps.")
    parser.add_argument("--kan_lbfgs_impl", type=str, default="torch", choices=["torch", "pykan"],
                        help="LBFGS backend to use for KAN training.")
    parser.add_argument("--kan_lbfgs_tolerance_ys", type=float, default=1e-32,
                        help="Curvature threshold for pykan LBFGS memory updates (ignored for torch LBFGS).")
    parser.add_argument("--kan_lamb_l1", type=float, default=1.0)
    parser.add_argument("--kan_lamb_entropy", type=float, default=2.0)
    parser.add_argument(
        "--kan_sparse_init",
        type=lambda x: str(x).strip().lower() in {"1", "true", "yes", "y", "on"},
        default=True,
        help="Use sparse initialization in pykan KANLayer connections (default: true).",
    )

    parser.add_argument("--kan_adam_warmup_epochs", type=int, default=10)
    parser.add_argument("--kan_grid_update_freq", type=int, default=10)
    parser.add_argument("--kan_grid_update_warmup", type=int, default=5)
    parser.add_argument("--kan_max_grid_updates", type=int, default=4)
    parser.add_argument("--kan_gradient_clip", type=float, default=1.0,
                        help="Gradient clipping value applied during Adam warmup; set to 0 or negative to disable.")
    parser.add_argument(
        "--kan_lamb_schedule",
        type=str,
        default=None,
        help=(
            "Optional JSON list for staged lamb values, "
            "e.g. '[1e-4, 1e-3, 5e-3, 1e-2]'. "
            "Applied in equal epoch segments for Graph-KAN training."
        ),
    )
    parser.add_argument("--kan_square_loss", action="store_true", default=False,
                        help="Use MSE (square=True) instead of L1 for KAN training.")
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--kan_batch_size", type=int, default=16)
    parser.add_argument(
        "--kan_adam_batch_size",
        type=int,
        default=None,
        help="Batch size for Adam warmup (defaults to kan_batch_size).",
    )
    parser.add_argument(
        "--kan_lbfgs_batch_size",
        type=int,
        default=None,
        help="Batch size for LBFGS phase (defaults to kan_batch_size).",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lamb", type=float, default=0)
    parser.add_argument(
        "--train-baseline",
        dest="train_baseline",
        action="store_true",
        default=True,
        help="Train the OGN baseline model.",
    )
    parser.add_argument(
        "--no-train-baseline",
        dest="train_baseline",
        action="store_false",
        help="Skip OGN baseline training.",
    )


    return parser.parse_args(args)



def _coerce_width_arg(value):
    """Convert CLI/YAML width input into a python list if provided."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Width must be JSON-like, e.g. '[10,[8,1],4]'"
            ) from exc
    return value


def _coerce_mult_arity_arg(value, label: str):
    """Convert CLI/YAML mult_arity into int or nested lists of ints."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be int or list, got boolean {value!r}.")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not float(value).is_integer():
            raise ValueError(f"{label} must be an integer or list-of-lists; got {value!r}.")
        return int(value)
    if isinstance(value, (list, tuple)):
        return [_coerce_mult_arity_arg(v, f"{label}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{label} must be int or JSON list-of-lists, e.g. 2 or '[[],[2,3],[]]'."
            ) from exc
        return _coerce_mult_arity_arg(parsed, label)
    raise ValueError(f"{label} must be int or list-of-lists, got {type(value).__name__}.")


def _coerce_float_schedule_arg(value, label: str) -> list[float] | None:
    """Convert CLI/YAML schedule input into a list[float]."""
    if value is None:
        return None

    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{label} must be a JSON list, e.g. '[1e-4, 1e-3, 5e-3, 1e-2]'."
            ) from exc

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of numeric values.")

    schedule: list[float] = []
    for i, item in enumerate(value):
        try:
            level = float(item)
        except Exception as exc:
            raise ValueError(f"{label}[{i}] must be numeric, got {item!r}.") from exc
        if level < 0.0:
            raise ValueError(f"{label}[{i}] must be >= 0, got {level}.")
        schedule.append(level)

    return schedule or None


def _extract_width_output_dim(width, label: str) -> int:
    """Extract total output dimension from pykan width spec."""
    if width is None:
        raise ValueError(f"{label} is required and must be a sequence.")
    if not isinstance(width, Sequence) or isinstance(width, (str, bytes)):
        raise ValueError(f"{label} must be a sequence, got {type(width).__name__}.")
    if len(width) < 2:
        raise ValueError(f"{label} must include at least input and output dimensions.")

    last = width[-1]
    if isinstance(last, Sequence) and not isinstance(last, (str, bytes)):
        if len(last) != 2:
            raise ValueError(
                f"{label} last entry must be a scalar or [linear, mult]; got {last!r}."
            )
        try:
            linear_out = int(last[0])
            mult_out = int(last[1])
        except Exception as exc:
            raise ValueError(
                f"{label} last entry must be numeric [linear, mult]; got {last!r}."
            ) from exc

        if linear_out < 0:
            raise ValueError(
                f"{label} output linear width must be >= 0; got {last!r}."
            )
        if mult_out < 0:
            raise ValueError(f"{label} output mult node count must be >= 0; got {last!r}.")
        total_out = linear_out + mult_out
        if total_out <= 0:
            raise ValueError(
                f"{label} output width must include at least one node; got {last!r}."
            )
        return total_out

    try:
        out_dim = int(last)
    except Exception as exc:
        raise ValueError(f"{label} must end with output dimension; got {last!r}.") from exc
    if out_dim <= 0:
        raise ValueError(f"{label} output dimension must be > 0; got {out_dim}.")
    return out_dim


def _coerce_feature_spec_arg(value):
    """Convert CLI/YAML feature spec input into dict/list form."""
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "input_features must be JSON dict/list, e.g. "
                "'{\"include\":[\"pos\",\"mass\"]}'"
            ) from exc
    raise ValueError(f"input_features must be dict/list/JSON-string, got {type(value).__name__}.")


def _align_width_input_dim(width, input_dim: int, label: str):
    """Ensure pykan width starts with expected input_dim; auto-fix for convenience."""
    if not isinstance(width, Sequence) or isinstance(width, (str, bytes)):
        raise ValueError(f"{label} must be a sequence, got {type(width).__name__}.")
    if len(width) < 2:
        raise ValueError(f"{label} must include at least input and output dimensions.")

    first = width[0]
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        raise ValueError(f"{label} first entry must be scalar input dimension, got {first!r}.")

    try:
        current = int(first)
    except Exception as exc:
        raise ValueError(f"{label} first entry must be numeric, got {first!r}.") from exc

    if current == int(input_dim):
        return width

    adjusted = list(width)
    adjusted[0] = int(input_dim)
    print(f"Adjusted {label}[0]: {current} -> {input_dim} to match selected node features.")
    return adjusted



def train_gnn(model, train_loader, val_loader, n_epochs, lr, device, lamb,
              augment=False, augmentation_scale=3.0):
    trainer = GNNTrainer(
        model, train_loader, val_loader,
        lr=lr, device=device,
        checkpoint_dir=None,
    )
    trainer.lamb = lamb
    trainer.train(
        n_epochs=n_epochs,
        augment=augment,
        augmentation_scale=augmentation_scale,
        save_every=n_epochs,
    )
    return model, trainer.history


def train_kan(model, train_loader, val_loader, n_epochs, device, lamb,
              adam_warmup_epochs: int = 0,
              adam_lr: float = 1e-3,
              grid_update_freq: int = 10,
              grid_update_warmup: int = 5,
              max_grid_updates: int = 4,
              lamb_schedule: list[float] | None = None,
              gradient_clip: float | None = 1.0,
              lbfgs_lr: float = 1.0,
              lbfgs_max_iter: int = 10,
              lbfgs_max_eval: int | None = None,
              lbfgs_line_search_fn: str | None = "strong_wolfe",
              lbfgs_impl: str = "torch",
              lbfgs_tolerance_ys: float = 1e-32,
              lbfgs_train_loader=None,
              square_loss: bool = False,
              augment: bool = False,
              augmentation_scale: float = 3.0):
    trainer = KANTrainer(
        model, train_loader, val_loader,
        lbfgs_train_loader=lbfgs_train_loader,
        lbfgs_lr=lbfgs_lr,
        lbfgs_max_iter=lbfgs_max_iter,
        lbfgs_max_eval=lbfgs_max_eval,
        lbfgs_line_search_fn=lbfgs_line_search_fn,
        lbfgs_impl=lbfgs_impl,
        lbfgs_tolerance_ys=lbfgs_tolerance_ys,
        adam_lr=adam_lr,
        adam_warmup_epochs=adam_warmup_epochs,
        device=device,
        checkpoint_dir=None,
        grid_update_freq=grid_update_freq,
        grid_update_warmup=grid_update_warmup,
        max_grid_updates=max_grid_updates,
        lamb_schedule=lamb_schedule,
    )
    trainer.lamb = lamb
    clip_value = None if (gradient_clip is None or gradient_clip <= 0) else gradient_clip
    trainer.train(
        n_epochs=n_epochs,
        augment=augment,
        augmentation_scale=augmentation_scale,
        save_every=n_epochs,
        gradient_clip=clip_value,
        square_loss=square_loss,
    )
    return model, trainer.history

def visualize_training_loss(
        history: dict[str, list[float]],
        save_path: str | Path | None = None,
        title: str = 'Training Loss'
) -> None:
    """
    Plot train and validation loss curves over epochs.

    Args:
        history:   dict with 'train' and 'val' loss lists
        save_path: if given, save figure here
        title:     plot title
    """
    epochs = range(len(history['train']))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history['train'], label='Train',      linewidth=2)
    ax.plot(epochs, history['val'],   label='Validation', linewidth=2, linestyle='--')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()

def main(yaml_params: Optional[dict] = None, checkpoint_dir: Optional[str] = None, data_dir: Optional[str] = None):
    args = parse_args([] if yaml_params is not None else None)

    if yaml_params is not None:
        args.train_data = str(Path(data_dir) / "train.npz")
        args.val_data = str(Path(data_dir) / "val.npz")
        args.include_velocity = yaml_params.get("include_velocity", args.include_velocity)
        args.input_features = yaml_params.get("input_features", args.input_features)
        args.checkpoint_dir = checkpoint_dir if checkpoint_dir is not None else args.checkpoint_dir
        args.hidden = yaml_params.get("gnn_hp", {}).get("hidden", args.hidden)
        args.msg_dim = yaml_params.get("gnn_hp", {}).get("msg_dim", args.msg_dim)
        args.kan_msg_width             = yaml_params.get("gkan_hp", {}).get("msg_width",            args.kan_msg_width)
        args.kan_node_width            = yaml_params.get("gkan_hp", {}).get("node_width",           args.kan_node_width)
        args.kan_msg_dim               = yaml_params.get("gkan_hp", {}).get("msg_dim",              args.kan_msg_dim)
        args.kan_msg_mult_arity        = yaml_params.get("gkan_hp", {}).get("msg_mult_arity",       args.kan_msg_mult_arity)
        args.kan_node_mult_arity       = yaml_params.get("gkan_hp", {}).get("node_mult_arity",      args.kan_node_mult_arity)
        args.kan_grid_size             = yaml_params.get("gkan_hp", {}).get("grid_size",            args.kan_grid_size)
        args.kan_base_fun              = yaml_params.get("gkan_hp", {}).get("base_fun",             args.kan_base_fun)
        args.kan_noise_scale           = yaml_params.get("gkan_hp", {}).get("noise_scale",          args.kan_noise_scale)
        args.kan_scale_base_mu         = yaml_params.get("gkan_hp", {}).get("scale_base_mu",        args.kan_scale_base_mu)
        args.kan_scale_base_sigma      = yaml_params.get("gkan_hp", {}).get("scale_base_sigma",     args.kan_scale_base_sigma)
        args.kan_lbfgs_lr              = yaml_params.get("gkan_hp", {}).get("lbfgs_lr",            args.kan_lbfgs_lr)
        args.kan_lbfgs_max_iter        = yaml_params.get("gkan_hp", {}).get("lbfgs_max_iter",      args.kan_lbfgs_max_iter)
        args.kan_lbfgs_max_eval        = yaml_params.get("gkan_hp", {}).get("lbfgs_max_eval",      args.kan_lbfgs_max_eval)
        args.kan_lbfgs_line_search_fn  = yaml_params.get("gkan_hp", {}).get("lbfgs_line_search_fn", args.kan_lbfgs_line_search_fn)
        args.kan_lbfgs_impl            = yaml_params.get("gkan_hp", {}).get("lbfgs_impl",          args.kan_lbfgs_impl)
        args.kan_lbfgs_tolerance_ys    = yaml_params.get("gkan_hp", {}).get("lbfgs_tolerance_ys",  args.kan_lbfgs_tolerance_ys)
        args.kan_lamb_l1               = yaml_params.get("gkan_hp", {}).get("lamb_l1",              args.kan_lamb_l1)
        args.kan_lamb_entropy          = yaml_params.get("gkan_hp", {}).get("lamb_entropy",         args.kan_lamb_entropy)
        args.kan_sparse_init           = yaml_params.get("gkan_hp", {}).get("sparse_init",          args.kan_sparse_init)
        args.kan_adam_warmup_epochs    = yaml_params.get("gkan_hp", {}).get("adam_warmup_epochs",   args.kan_adam_warmup_epochs)
        args.kan_grid_update_freq      = yaml_params.get("gkan_hp", {}).get("grid_update_freq",     args.kan_grid_update_freq)
        args.kan_grid_update_warmup    = yaml_params.get("gkan_hp", {}).get("grid_update_warmup",   args.kan_grid_update_warmup)
        args.kan_max_grid_updates      = yaml_params.get("gkan_hp", {}).get("max_grid_updates",     args.kan_max_grid_updates)
        args.kan_gradient_clip         = yaml_params.get("gkan_hp", {}).get("gradient_clip",        args.kan_gradient_clip)
        args.kan_square_loss           = yaml_params.get("training_hp", {}).get("kan_square_loss",  args.kan_square_loss)
        args.kan_lamb_schedule         = yaml_params.get("training_hp", {}).get("kan_lamb_schedule", args.kan_lamb_schedule)
        args.epochs = yaml_params.get("training_hp", {}).get("epochs", args.epochs)
        args.batch_size = yaml_params.get("training_hp", {}).get("batch_size", args.batch_size)
        args.kan_batch_size = yaml_params.get("training_hp", {}).get("kan_batch_size", args.kan_batch_size)
        args.kan_adam_batch_size = yaml_params.get("training_hp", {}).get(
            "kan_adam_batch_size", args.kan_adam_batch_size
        )
        args.kan_lbfgs_batch_size = yaml_params.get("training_hp", {}).get(
            "kan_lbfgs_batch_size", args.kan_lbfgs_batch_size
        )
        args.lr = float(yaml_params.get("training_hp", {}).get("lr", args.lr))
        args.lamb = yaml_params.get("training_hp", {}).get("lamb", args.lamb)
        args.augment = yaml_params.get("training_hp", {}).get("augment", False)
        args.augmentation_scale = yaml_params.get("training_hp", {}).get("augmentation_scale", 3.0)
        args.train_baseline = yaml_params.get("train_baseline", args.train_baseline)

    args.kan_msg_width = _coerce_width_arg(args.kan_msg_width)
    args.kan_node_width = _coerce_width_arg(args.kan_node_width)
    args.input_features = _coerce_feature_spec_arg(args.input_features)
    args.kan_msg_mult_arity = _coerce_mult_arity_arg(args.kan_msg_mult_arity, "kan_msg_mult_arity")
    args.kan_node_mult_arity = _coerce_mult_arity_arg(args.kan_node_mult_arity, "kan_node_mult_arity")
    args.kan_lamb_schedule = _coerce_float_schedule_arg(
        args.kan_lamb_schedule,
        "kan_lamb_schedule",
    )

    if args.kan_adam_batch_size is None:
        args.kan_adam_batch_size = args.kan_batch_size
    if args.kan_lbfgs_batch_size is None:
        args.kan_lbfgs_batch_size = args.kan_batch_size

    # Infer msg_dim from width if provided to avoid mismatch with pykan spec.
    if not args.kan_msg_width or not args.kan_node_width:
        raise ValueError("kan_msg_width and kan_node_width are required (pykan width format).")

    msg_out = _extract_width_output_dim(args.kan_msg_width, "kan_msg_width")
    if args.kan_msg_dim is None:
        args.kan_msg_dim = msg_out
    elif args.kan_msg_dim != msg_out:
        raise ValueError(f"kan_msg_dim ({args.kan_msg_dim}) must match kan_msg_width output ({msg_out}).")

    feature_spec = normalize_feature_spec(
        feature_spec=args.input_features,
        include_velocity=(args.include_velocity if args.input_features is None else None),
    )

    device = get_device()
    print(f"Using device: {device}\n")

    # Load data
    print("Loading datasets...")
    train_dataset = NBodyDataset(args.train_data, feature_spec=feature_spec)
    val_dataset = NBodyDataset(args.val_data, feature_spec=feature_spec)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    kan_adam_train_loader = DataLoader(train_dataset, batch_size=args.kan_adam_batch_size, shuffle=True, num_workers=8, pin_memory=False, persistent_workers=True)
    if args.kan_lbfgs_batch_size == args.kan_adam_batch_size:
        kan_lbfgs_train_loader = kan_adam_train_loader
    else:
        kan_lbfgs_train_loader = DataLoader(train_dataset, batch_size=args.kan_lbfgs_batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    kan_val_loader = DataLoader(val_dataset, batch_size=args.kan_lbfgs_batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    print(
        "Loader steps/epoch: "
        f"GNN={len(train_loader)} (batch={args.batch_size}), "
        f"Graph-KAN Adam={len(kan_adam_train_loader)} (batch={args.kan_adam_batch_size}), "
        f"LBFGS={len(kan_lbfgs_train_loader)} (batch={args.kan_lbfgs_batch_size})"
    )
    if args.kan_lamb_schedule:
        print(
            "Graph-KAN lamb schedule (uniform ramp): "
            + ", ".join(f"{v:.2e}" for v in args.kan_lamb_schedule)
        )
    if len(kan_lbfgs_train_loader) <= 1:
        print(
            "Warning: Graph-KAN LBFGS phase is effectively full-batch (<=1 step/epoch). "
            "If convergence per epoch is slow, reduce kan_lbfgs_batch_size."
        )

    n_features = train_dataset.n_node_features
    args.kan_msg_width = _align_width_input_dim(args.kan_msg_width, 2 * n_features, "kan_msg_width")
    args.kan_node_width = _align_width_input_dim(
        args.kan_node_width,
        n_features + int(args.kan_msg_dim),
        "kan_node_width",
    )
    edge_index = train_dataset.edge_index

    if args.kan_node_width:
        expected_out = train_dataset.dim
        node_out = _extract_width_output_dim(args.kan_node_width, "kan_node_width")
        if node_out != expected_out:
            raise ValueError(
                f"kan_node_width last element must equal dataset dim ({expected_out}), got {node_out}."
            )

    print(f"Dataset: {train_dataset.n} bodies, {train_dataset.dim}D")
    print(
        "Input feature spec: "
        f"include={feature_spec['include']}, augment={feature_spec['augment']}"
    )
    print(f"Node features: {n_features}")
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples\n")

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create and train Graph-KAN
    print("="*60)
    print("Graph-KAN")
    print("="*60)
    kan_model = OrdinaryGraphKAN(
        n_f=n_features,
        msg_width=args.kan_msg_width,
        node_width=args.kan_node_width,
        edge_index=edge_index,
        input_feature_spec=feature_spec,
        grid_size=args.kan_grid_size,
        spline_order=3,
        aggr="add",
        lamb_l1=args.kan_lamb_l1,
        lamb_entropy=args.kan_lamb_entropy,
        sparse_init=args.kan_sparse_init,
        msg_mult_arity=args.kan_msg_mult_arity,
        node_mult_arity=args.kan_node_mult_arity,
        base_fun=args.kan_base_fun,
        noise_scale=args.kan_noise_scale,
        scale_base_mu=args.kan_scale_base_mu,
        scale_base_sigma=args.kan_scale_base_sigma,
    )
    kan_model.summary()
    print(" ")

    kan_model, kan_history = train_kan(
        kan_model, kan_adam_train_loader, kan_val_loader,
        n_epochs=args.epochs,
        device=device,
        lamb=args.lamb,
        adam_warmup_epochs=args.kan_adam_warmup_epochs,
        adam_lr=args.lr,
        grid_update_freq=args.kan_grid_update_freq,
        grid_update_warmup=args.kan_grid_update_warmup,
        max_grid_updates=args.kan_max_grid_updates,
        lamb_schedule=args.kan_lamb_schedule,
        gradient_clip=args.kan_gradient_clip,
        lbfgs_lr=args.kan_lbfgs_lr,
        lbfgs_max_iter=args.kan_lbfgs_max_iter,
        lbfgs_max_eval=args.kan_lbfgs_max_eval,
        lbfgs_line_search_fn=args.kan_lbfgs_line_search_fn,
        lbfgs_impl=args.kan_lbfgs_impl,
        lbfgs_tolerance_ys=args.kan_lbfgs_tolerance_ys,
        lbfgs_train_loader=kan_lbfgs_train_loader,
        square_loss=args.kan_square_loss,
        augment=args.augment,
        augmentation_scale=args.augmentation_scale,
    )
    visualize_training_loss(kan_history,
                            title='Graph-KAN Training Loss',
                            save_path=f'{checkpoint_dir}/kan_loss.png')
    gkan_checkpoint_path = f"{checkpoint_dir}/graph_kan.pt"
    loader = ModelLoader(OrdinaryGraphKAN, gkan_checkpoint_path)
    loader.save(kan_model, gkan_checkpoint_path)
    print(f"Saved checkpoint: {gkan_checkpoint_path}\n")

    if args.train_baseline:
        # Create and train Baseline GNN
        print("="*60)
        print("Baseline GNN")
        print("="*60)
        gnn_model = OGN(
            n_f=n_features, msg_dim=args.msg_dim, ndim=train_dataset.dim,
            edge_index=edge_index, hidden=args.hidden, aggr="add",
            input_feature_spec=feature_spec,
        )
        gnn_model.summary()
        print(" ")

        gnn_model, gnn_history = train_gnn(
            gnn_model,
            train_loader,
            val_loader,
            args.epochs,
            args.lr,
            device,
            args.lamb,
            args.augment,
            args.augmentation_scale,
        )
        visualize_training_loss(gnn_history,
                                title='GNN Training Loss',
                                save_path=f'{checkpoint_dir}/gnn_loss.png')

        gnn_checkpoint_path = f"{checkpoint_dir}/baseline_gnn.pt"
        loader = ModelLoader(OGN, gnn_checkpoint_path)
        loader.save(gnn_model, gnn_checkpoint_path)
        print(f"Saved checkpoint: {gnn_checkpoint_path}\n")
    else:
        print("="*60)
        print("Baseline GNN")
        print("="*60)
        print("Skipping baseline training (train_baseline=False).\n")
    print("="*60)
    print("Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
