"""Per-layer symbolic regression + analytic composition for GKAN.

For each trained model:
1. Registers physics-relevant candidates (including x^{-3/2} for r^2 -> r^{-3})
2. Runs pykan's suggest_symbolic() per-edge with low complexity weight
   to prioritize R^2 (avoids prematurely zeroing active edges)
3. Fixes all edges to their symbolic functions with affine parameter fitting
4. Uses pykan's symbolic_formula() to compose across layers
5. Also builds a manual composition (dropping small coefficients) as a cleaner alternative
6. Generates per-layer plots: actual spline postacts vs symbolic fit overlay
"""

import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kan.utils import SYMBOLIC_LIB
from nbody_gkan.models import OrdinaryGraphKAN
from nbody_gkan.models.model_loader import ModelLoader

EXPERIMENTS = ["small_rel_pos_idc", "small_rel_pos_dsq", "small_rel_pos", "small"]
CHECKPOINT_DIR = Path("checkpoints")
DATA_DIR = Path("data")
OUTPUT_DIR = Path("interpretability")
R2_THRESHOLD = 0.85
N_PLOT = 200
COEFF_DEAD = 0.01  # |c| below this → treat edge as dead


# ── Register custom symbolic functions ────────────────────────────

def register_custom_symbols():
    if "x^(-1.5)" not in SYMBOLIC_LIB:
        SYMBOLIC_LIB["x^(-1.5)"] = (
            lambda x: torch.where(x > 0.01, x.pow(-1.5), torch.zeros_like(x)),
            lambda x: x ** sympy.Rational(-3, 2),
            3,
            lambda x, y_th: (x > 0.01,),
        )
    if "x^(-0.5)" not in SYMBOLIC_LIB:
        SYMBOLIC_LIB["x^(-0.5)"] = (
            lambda x: torch.where(x > 0.01, x.pow(-0.5), torch.zeros_like(x)),
            lambda x: x ** sympy.Rational(-1, 2),
            2,
            lambda x, y_th: (x > 0.01,),
        )


CANDIDATE_LIB = [
    "0", "x", "x^2", "x^3",
    "1/x", "1/x^2",
    "abs",
    "x^(-1.5)", "x^(-0.5)",
]


# ── Feature names ─────────────────────────────────────────────────

def msg_input_names(model):
    base = ["px", "py", "vx", "vy", "m"][: model.n_f]
    names = [f"{v}_i" for v in base] + [f"{v}_j" for v in base]
    for feat in model.edge_augmentations or []:
        if feat == "rel_pos":
            names += ["dx", "dy"]
        elif feat == "dist_sq":
            names += ["r_sq"]
        elif feat == "inv_dist_cu":
            names += ["idc"]
    return names


def node_input_names(model):
    base = ["px", "py", "vx", "vy", "m"][: model.n_f]
    return base + [f"msg_{i}" for i in range(model.msg_dim)]


# ── Data helpers ──────────────────────────────────────────────────

def build_graph_loader(model, data_path, n_samples=200, batch_size=10):
    data = np.load(data_path)
    masses = torch.from_numpy(data["masses"]).float()
    rng = np.random.default_rng(42)
    n = min(n_samples, data["positions"].shape[0])
    idx = rng.choice(data["positions"].shape[0], n, replace=False)
    graphs = []
    for i in idx:
        pos = torch.from_numpy(data["positions"][i]).float()
        vel = torch.from_numpy(data["velocities"][i]).float()
        x = torch.cat([pos, vel, masses.unsqueeze(1)], dim=1)
        graphs.append(Data(x=x, edge_index=model.edge_index))
    return PyGLoader(graphs, batch_size=batch_size)


# ── Core: suggest + fix in one pass ───────────────────────────────

def suggest_and_fix(model, loader, lib, max_batches=10, weight_simple=0.1):
    """Populate activations, suggest symbolic per-edge, fix immediately.

    Uses low weight_simple (default 0.1) to prioritize R^2 over simplicity,
    so active edges aren't aggressively zeroed out.
    """
    model.eval()
    model.msg_kan.symbolic_enabled = False
    model.node_kan.symbolic_enabled = False

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            model.propagate(
                batch.edge_index,
                size=(batch.x.size(0), batch.x.size(0)),
                x=batch.x,
            )

    results = {}
    for tag, kan in [("msg", model.msg_kan), ("node", model.node_kan)]:
        layers = {}
        for l, layer in enumerate(kan.act_fun):
            edges = {}
            for i in range(layer.in_dim):
                for j in range(layer.out_dim):
                    try:
                        fn, _, r2, c = kan.suggest_symbolic(
                            l, i, j, lib=lib, topk=1, verbose=False,
                            weight_simple=weight_simple,
                        )
                    except Exception:
                        fn, r2, c = "0", 0.0, 0

                    # Override to zero if R² is too low
                    r2 = float(r2)
                    if fn != "0" and r2 < 0.5:
                        fn = "0"
                        r2 = 0.0

                    try:
                        kan.fix_symbolic(
                            l, i, j, fn,
                            fit_params_bool=True, verbose=False, log_history=False,
                        )
                        affine = kan.symbolic_fun[l].affine[j, i].detach().tolist()
                    except Exception:
                        affine = [1.0, 0.0, 0.0, 0.0]

                    edges[(i, j)] = {
                        "fn": fn,
                        "r2": r2,
                        "affine": affine,  # [a_x, b_x, c, d]
                    }
            layers[l] = edges
        results[tag] = layers
    return results


def unfix_all(kan, layer_results):
    for l, edges in layer_results.items():
        for (i, j) in edges:
            try:
                kan.unfix_symbolic(l, i, j, log_history=False)
            except Exception:
                pass


# ── pykan composition ─────────────────────────────────────────────

def compose_pykan(kan, var_names):
    """Use pykan's symbolic_formula for the official composition."""
    import signal

    var = [sympy.Symbol(n) for n in var_names]
    try:
        formulas, _ = kan.symbolic_formula(var=var)
    except Exception as e:
        return None, str(e)

    simplified = []
    for f in formulas:
        try:
            s = sympy.nsimplify(f, rational=True, tolerance=0.05)
            # Timeout simplify at 10s — deep networks produce huge expressions
            old_handler = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError))
            signal.alarm(10)
            try:
                s = sympy.simplify(s)
            except (TimeoutError, Exception):
                pass
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        except Exception:
            s = f
        simplified.append(s)
    return simplified, None


# ── Manual composition (cleaner) ──────────────────────────────────

def compose_manual(layer_results, var_names, out_names):
    """Build sympy expressions by hand from the per-edge fits.

    Drops edges where |c| < COEFF_DEAD and rounds coefficients to simple
    rationals so the composed expression is more readable.
    """
    current = [sympy.Symbol(n) for n in var_names]

    for l in sorted(layer_results.keys()):
        edges = layer_results[l]
        in_dim = max(i for (i, _) in edges) + 1
        out_dim = max(j for (_, j) in edges) + 1

        next_layer = []
        for j in range(out_dim):
            expr = sympy.Integer(0)
            const_total = 0.0
            for i in range(in_dim):
                info = edges.get((i, j))
                if not info or info["fn"] == "0":
                    continue
                a_x, b_x, c, d = info["affine"]
                if abs(c) < COEFF_DEAD:
                    const_total += d
                    continue

                fn_name = info["fn"]
                if fn_name not in SYMBOLIC_LIB:
                    continue
                sp_fn = SYMBOLIC_LIB[fn_name][1]

                c_r = sympy.nsimplify(c, rational=True, tolerance=0.05)
                a_r = sympy.nsimplify(a_x, rational=True, tolerance=0.05)
                b_r = sympy.nsimplify(b_x, rational=True, tolerance=0.05)
                d_r = sympy.nsimplify(d, rational=True, tolerance=0.05)

                inner = a_r * current[i] + b_r if abs(b_x) > 0.01 else a_r * current[i]
                if abs(a_x - 1.0) < 0.05 and abs(b_x) < 0.01:
                    inner = current[i]
                term = c_r * sp_fn(inner) + d_r
                expr += term

            if abs(const_total) > 0.01:
                expr += sympy.nsimplify(const_total, rational=True, tolerance=0.05)
            next_layer.append(expr)

        current = next_layer

    import signal

    result = {}
    for j, name in enumerate(out_names):
        raw = current[j]
        try:
            old_handler = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError))
            signal.alarm(10)
            try:
                s = sympy.simplify(sympy.expand(raw))
            except (TimeoutError, Exception):
                s = raw
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        except Exception:
            s = raw
        result[name] = s
    return result


# ── Plotting ──────────────────────────────────────────────────────

def plot_spline_fits(kan, layer_results, input_names, tag, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for l, edges in layer_results.items():
        layer = kan.act_fun[l]
        in_dim, out_dim = layer.in_dim, layer.out_dim

        if not hasattr(kan, "acts") or l >= len(kan.acts):
            continue
        acts_in = kan.acts[l]
        if acts_in.dim() < 2 or acts_in.shape[1] < in_dim:
            continue

        postacts = None
        if hasattr(layer, "postacts") and layer.postacts is not None:
            postacts = layer.postacts.detach().cpu()

        fig, axes = plt.subplots(
            out_dim, in_dim,
            figsize=(max(2.2 * in_dim, 6), 2.5 * out_dim),
            squeeze=False,
        )
        fig.suptitle(f"{tag} KAN — Layer {l} ({in_dim}→{out_dim})", fontsize=11)

        for i in range(in_dim):
            x_data = acts_in[:, i].detach().cpu()
            lo, hi = float(x_data.quantile(0.02)), float(x_data.quantile(0.98))
            if hi - lo < 0.3:
                mid = (lo + hi) / 2
                lo, hi = mid - 0.5, mid + 0.5
            x_plot = torch.linspace(lo, hi, N_PLOT)

            for j in range(out_dim):
                ax = axes[j, i]
                info = edges.get((i, j), {})
                fn_name = info.get("fn", "?")
                r2 = info.get("r2", 0.0)
                affine = info.get("affine", [1, 0, 1, 0])

                if postacts is not None and postacts.shape[2] > i and postacts.shape[1] > j:
                    x_pts = x_data.numpy()
                    y_pts = postacts[:, j, i].numpy()
                    n = min(500, len(x_pts))
                    idx = np.random.default_rng(0).choice(len(x_pts), n, replace=False)
                    ax.scatter(x_pts[idx], y_pts[idx], s=1, alpha=0.3, c="steelblue")

                if fn_name != "0" and fn_name in SYMBOLIC_LIB:
                    a_x, b_x, c, d = affine
                    torch_fn = SYMBOLIC_LIB[fn_name][0]
                    try:
                        y_fit = c * torch_fn(a_x * x_plot + b_x) + d
                        ax.plot(x_plot.numpy(), y_fit.numpy(), "r-", linewidth=1, alpha=0.8)
                    except Exception:
                        pass

                color = "green" if r2 >= R2_THRESHOLD else ("orange" if r2 > 0.5 else "gray")
                lbl = f"{fn_name} R²={r2:.2f}"
                if fn_name != "0":
                    _, _, c_coeff, _ = affine
                    lbl += f" c={c_coeff:.2f}"
                ax.set_title(lbl, fontsize=6, color=color, pad=2)

                if l == 0:
                    src = input_names[i] if i < len(input_names) else f"x{i}"
                else:
                    src = f"h{l-1}_{i}"
                if j == out_dim - 1:
                    ax.set_xlabel(src, fontsize=7)
                ax.tick_params(labelsize=5)
                ax.grid(True, alpha=0.2)

        plt.tight_layout()
        fig.savefig(save_dir / f"{tag}_layer{l}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {save_dir / f'{tag}_layer{l}.png'}")


# ── Per-layer summary ─────────────────────────────────────────────

def print_layer_summary(layer_results, input_names, tag):
    print(f"\n  {tag} KAN per-layer fits:")
    for l, edges in sorted(layer_results.items()):
        in_dim = max(i for (i, _) in edges) + 1
        out_dim = max(j for (_, j) in edges) + 1
        print(f"    Layer {l} ({in_dim}→{out_dim}):")

        for j in range(out_dim):
            parts = []
            for i in range(in_dim):
                info = edges.get((i, j), {})
                fn = info.get("fn", "0")
                r2 = info.get("r2", 0.0)
                if fn == "0":
                    continue
                a_x, b_x, c, d = info.get("affine", [1, 0, 1, 0])
                if abs(c) < COEFF_DEAD and abs(d) < COEFF_DEAD:
                    continue
                if l == 0:
                    src = input_names[i] if i < len(input_names) else f"x{i}"
                else:
                    src = f"h{l-1}_{i}"
                parts.append(f"{c:+.3f}·{fn}({a_x:.2f}·{src}{b_x:+.2f}){d:+.3f} R²={r2:.3f}")
            if parts:
                print(f"      h{l}_{j} = " + "\n             + ".join(parts))
            else:
                print(f"      h{l}_{j} = ZERO")


# ── Main ──────────────────────────────────────────────────────────

def analyze_model(name):
    ckpt_path = CHECKPOINT_DIR / name / "graph_kan.pt"
    data_path = DATA_DIR / name / "train.npz"
    if not ckpt_path.exists():
        print(f"  Skipping {name}: no checkpoint")
        return

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    model, _ = ModelLoader(OrdinaryGraphKAN, str(ckpt_path)).load()
    loader = build_graph_loader(model, data_path)
    msg_names = msg_input_names(model)
    node_names = node_input_names(model)

    print("  Suggest + fix symbolic (weight_simple=0.1)...")
    results = suggest_and_fix(model, loader, CANDIDATE_LIB, weight_simple=0.1)

    out_dir = OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    for tag, kan, var_names, out_names in [
        ("msg", model.msg_kan, msg_names, [f"msg_{i}" for i in range(model.msg_dim)]),
        ("node", model.node_kan, node_names, ["a_x", "a_y"]),
    ]:
        layer_results = results[tag]

        # Summary (affine params now available since we already fixed)
        print_layer_summary(layer_results, var_names, tag.title())

        # Plots
        plot_spline_fits(kan, layer_results, var_names, tag, out_dir)

        # pykan composition
        formulas, err = compose_pykan(kan, var_names)
        if err:
            print(f"\n  {tag} pykan composition error: {err}")
        elif formulas:
            print(f"\n  {tag} KAN composed (pykan):")
            for i, f in enumerate(formulas):
                print(f"    {out_names[i]} = {f}")

        # Manual composition
        manual = compose_manual(layer_results, var_names, out_names)
        if manual:
            print(f"\n  {tag} KAN composed (manual, cleaned):")
            for name_out, expr in manual.items():
                print(f"    {name_out} = {expr}")

        unfix_all(kan, layer_results)

    print(f"\n  Plots saved to {out_dir}/")


def main():
    register_custom_symbols()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in EXPERIMENTS:
        analyze_model(name)
    print(f"\nDone. All results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()