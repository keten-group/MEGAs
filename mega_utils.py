"""Shared utilities for the MEGAs machine-learning notebooks.

Data loading, the model classes used in the manuscript, the geometric IoU
metric for cluster reconstruction, the Palmer et al. uncertainty calibration,
and Pareto-set helpers.  Every notebook imports this module and reads only
from ``data/``.

Conventions
-----------
* Design inputs are always ``(L0Bar, L0Y)`` in cm.
* A train/test split is a :class:`Split` record of integer indices into the
  design table; train and test never overlap.
"""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results")
FIGURES_DIR = os.path.join(HERE, "figures")
MODELS_DIR = os.path.join(HERE, "models")
for _d in (RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
    os.makedirs(_d, exist_ok=True)

# the supplementary data files of the paper
CONFIG_XLSX = os.path.join(DATA_DIR, "Supplementary Data 4.xlsx")     # configuration type + cluster configurations
PROPERTY_XLSX = os.path.join(DATA_DIR, "Supplementary Data 3.xlsx")   # strength and toughness

# --------------------------------------------------------------------------
# Design-space constants
# --------------------------------------------------------------------------

L_BAR_BOUNDS = (1.0, 2.0)
L_Y_BOUNDS = (1.0, 2.5)

OUTPUTS = ("Strength", "Toughness")
OUTPUT_UNITS = {"Strength": "N/m", "Toughness": "J/m$^2$"}

INFEASIBLE_CLASS = 4          # expanded configuration: no property prediction
CLASS_NAMES = {1: "Bar-dominant", 2: "Balanced", 3: "Y-dominant", 4: "Expanded"}

# palette used in all figures
BLUE = "#64A1FF"
GREY = "#9C9C9C"


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


def load_classification() -> pd.DataFrame:
    """Configuration-type labels on the 11 x 16 grid of branch lengths.

    The sheet ``configuration type`` holds the labels as a table with
    ``L0,Bar`` across columns and ``L0,Y`` down rows (cell range C2:M18).
    Returns one row per design with ``L0Bar, L0Y, Class`` (1 Bar-dominant,
    2 balanced, 3 Y-dominant, 4 expanded).
    """
    from openpyxl import load_workbook

    ws = load_workbook(CONFIG_XLSX, data_only=True)["configuration type"]
    rows = [[c.value for c in r] for r in ws["C2:M18"]]
    table = np.array(rows[1:], dtype=float)                # 16 (L0Y) x 11 (L0Bar)
    bar = np.round(np.linspace(1.0, 2.0, 11), 1)
    yl = np.round(np.linspace(1.0, 2.5, 16), 1)
    out = [(bar[c], yl[r], int(table[r, c])) for c in range(len(bar)) for r in range(len(yl))]
    return pd.DataFrame(out, columns=["L0Bar", "L0Y", "Class"])


def load_replicates() -> pd.DataFrame:
    """Raw simulation replicates (several rows per design) in physical units."""
    df = pd.read_excel(PROPERTY_XLSX)
    df = df.rename(columns={"L0,Bar": "L0Bar", "L0,Y": "L0Y",
                            "Strength [N/m]": "Strength", "Toughness [J/m2]": "Toughness"})
    df["L0Bar"] = df["L0Bar"].round(2)
    df["L0Y"] = df["L0Y"].round(2)
    return df[["L0Bar", "L0Y", "Strength", "Toughness"]]


def load_points() -> pd.DataFrame:
    """The 118 contracted designs with replicate-averaged strength and toughness."""
    rep = load_replicates()
    out = rep.groupby(["L0Bar", "L0Y"], as_index=False).agg(
        Strength=("Strength", "mean"), Toughness=("Toughness", "mean"), n=("Strength", "size"))
    out = out.merge(load_classification(), on=["L0Bar", "L0Y"], how="left")
    return out.sort_values(["L0Bar", "L0Y"]).reset_index(drop=True)


def xy(points: pd.DataFrame):
    X = points[["L0Bar", "L0Y"]].to_numpy(dtype=np.float64)
    Y = points[list(OUTPUTS)].to_numpy(dtype=np.float64)
    return X, Y


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def r2(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return np.nan if ss_tot == 0.0 else 1.0 - ss_res / ss_tot


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Split:
    protocol: str
    fold: str
    train: np.ndarray
    test: np.ndarray

    def __post_init__(self) -> None:
        assert len(np.intersect1d(self.train, self.test)) == 0, f"{self.fold}: train/test overlap"


def repeated_random(n: int, n_repeats: int = 20, test_size: float = 0.2) -> list[Split]:
    """Random 80/20 training/testing splits with seeds 0 .. n_repeats-1."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(n)
    out = []
    for rep in range(n_repeats):
        tr, te = train_test_split(idx, test_size=test_size, random_state=rep)
        out.append(Split("repeated_random", f"seed{rep:02d}", tr, te))
    return out


def property_fixed_split(n: int):
    """The fixed 70 / 24 / 24 training / validation / testing split used for the
    NN architecture search and for the deployed ensemble."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(n)
    idx_trainval, idx_test = train_test_split(idx, test_size=0.2, random_state=42)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=0.25, random_state=24)
    return idx_train, idx_val, idx_test


# --------------------------------------------------------------------------
# Result IO
# --------------------------------------------------------------------------


def save_results(df: pd.DataFrame, name: str) -> str:
    path = os.path.join(RESULTS_DIR, name)
    df.to_csv(path, index=False)
    print(f"wrote results/{name}  ({len(df)} rows)")
    return path


def load_results(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS_DIR, name))


def save_figure(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES_DIR, f"{name}.{ext}"), bbox_inches="tight", dpi=300)
    print(f"wrote figures/{name}.pdf/.png")


# --------------------------------------------------------------------------
# Cluster-configuration data
# --------------------------------------------------------------------------

CONFIG_SHEETS = {"Bar-dominant": "1 Bar-dominant", "Balanced": "2 Balanced", "Y-dominant": "3 Y-dominant"}
CONFIG_TASKS = ("xB", "xY", "yY", "thetaB", "thetaY")
CONFIG_TASK_UNITS = {"xB": "cm", "xY": "cm", "yY": "cm", "thetaB": "rad", "thetaY": "rad"}


def _rot(x, y, phi):
    c, s = np.cos(phi), np.sin(phi)
    return c * x - s * y, s * x + c * y


def load_configuration(sheet_label: str) -> pd.DataFrame:
    """One configuration type: duplicate representations of a design are
    averaged, then each configuration is rotated so that the reference Bar
    unit lies on the x-axis (YBar = 0), leaving five free parameters."""
    df = pd.read_excel(CONFIG_XLSX, sheet_name=CONFIG_SHEETS[sheet_label])
    a = df.groupby(["L0,Bar [cm]", "L0,Y [cm]"], as_index=False).mean()
    a = a.rename(columns={"L0,Bar [cm]": "L0Bar", "L0,Y [cm]": "L0Y",
                          "XBar [cm]": "xB", "YBar [cm]": "yB", "XY [cm]": "xY", "YY [cm]": "yY",
                          "θBar [rad]": "thetaB", "θY [rad]": "thetaY"})
    rows = []
    for i in range(len(a)):
        phi = -np.arctan2(a["yB"][i], a["xB"][i])
        xB, yB = _rot(a["xB"][i], a["yB"][i], phi)
        xY, yY = _rot(a["xY"][i], a["yY"][i], phi)
        rows.append({"L0Bar": round(float(a["L0Bar"][i]), 1), "L0Y": round(float(a["L0Y"][i]), 1),
                     "xB": xB, "yB": yB, "xY": xY, "yY": yY,
                     "thetaB": a["thetaB"][i] + phi, "thetaY": a["thetaY"][i] + phi})
    return pd.DataFrame(rows).reset_index(drop=True)


def load_configuration_all() -> pd.DataFrame:
    """All 118 contracted designs pooled, with a ``Type`` column.  θB has a π
    ambiguity (a bar is symmetric) and is wrapped to (-π/2, π/2]."""
    frames = []
    for name in CONFIG_SHEETS:
        d = load_configuration(name)
        d["Type"] = name
        frames.append(d)
    data = pd.concat(frames, ignore_index=True)
    data["thetaB"] = wrap_pi(data["thetaB"])
    assert data.duplicated(subset=["L0Bar", "L0Y"]).sum() == 0
    return data


def wrap_pi(t):
    return ((np.asarray(t, dtype=float) + np.pi / 2) % np.pi) - np.pi / 2


def config_xy(df: pd.DataFrame):
    X = df[["L0Bar", "L0Y"]].to_numpy(dtype=np.float64)
    Y = df[list(CONFIG_TASKS)].to_numpy(dtype=np.float64)
    return X, Y


# --------------------------------------------------------------------------
# Multi-output Gaussian process (GPyTorch)
# --------------------------------------------------------------------------


class MultitaskGPSurrogate:
    """Multi-output GP with an intrinsic-coregionalisation kernel: constant
    mean, an RBF kernel wrapped in a scale kernel, and a rank-1
    coregionalisation matrix.  Task-specific noise is pinned to 1e-6 so the
    model interpolates the (averaged) observations."""

    def __init__(self, iters: int = 300, lr: float = 0.1, seed: int = 42) -> None:
        self.iters, self.lr, self.seed = iters, lr, seed

    def _build(self, tx, ty):
        import gpytorch
        from gpytorch.models import ExactGP
        from gpytorch.means import MultitaskMean, ConstantMean
        from gpytorch.constraints import GreaterThan
        from gpytorch.kernels import ScaleKernel, RBFKernel, MultitaskKernel
        from gpytorch.likelihoods import MultitaskGaussianLikelihood
        from gpytorch.distributions import MultitaskMultivariateNormal
        import torch

        class _Model(ExactGP):
            def __init__(self, tx, ty, lik, nt):
                super().__init__(tx, ty, lik)
                self.mean_module = MultitaskMean(ConstantMean(), num_tasks=nt)
                self.covar_module = MultitaskKernel(ScaleKernel(RBFKernel()), num_tasks=nt, rank=1)

            def forward(self, x):
                return MultitaskMultivariateNormal(self.mean_module(x), self.covar_module(x))

        nt = ty.shape[1]
        lik = MultitaskGaussianLikelihood(num_tasks=nt, noise_constraint=GreaterThan(1e-9))
        lik.task_noises = torch.full((nt,), 1e-6)
        lik.raw_task_noises.requires_grad_(False)
        return _Model(tx, ty, lik, nt), lik

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "MultitaskGPSurrogate":
        import torch
        import gpytorch

        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        tx = torch.from_numpy(np.asarray(X, np.float32))
        ty = torch.from_numpy(np.asarray(Y, np.float32))
        model, lik = self._build(tx, ty)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
        for _ in range(self.iters):
            opt.zero_grad()
            loss = -mll(model(tx), ty)
            loss.backward()
            opt.step()
        model.eval(); lik.eval()
        self.model_, self.lik_, self.train_x_, self.train_y_ = model, lik, tx, ty
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        import torch

        with torch.no_grad():
            d = self.lik_(self.model_(torch.from_numpy(np.asarray(X, np.float32))))
            mu = d.mean.numpy().astype(np.float64)
            if not return_std:
                return mu
            return mu, np.sqrt(d.variance.numpy().astype(np.float64))

    def save(self, path: str) -> None:
        import torch
        torch.save({"state_dict": self.model_.state_dict(), "train_x": self.train_x_,
                    "train_y": self.train_y_, "iters": self.iters, "lr": self.lr, "seed": self.seed}, path)

    @classmethod
    def load(cls, path: str) -> "MultitaskGPSurrogate":
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self = cls(iters=ck["iters"], lr=ck["lr"], seed=ck["seed"])
        model, lik = self._build(ck["train_x"], ck["train_y"])
        model.load_state_dict(ck["state_dict"])
        model.eval(); lik.eval()
        self.model_, self.lik_, self.train_x_, self.train_y_ = model, lik, ck["train_x"], ck["train_y"]
        return self


# --------------------------------------------------------------------------
# Neural network for property prediction (PyTorch)
# --------------------------------------------------------------------------


def build_mlp(hidden_sizes, activation="Sigmoid", dropout=0.0):
    """Fully connected network 2 -> hidden_sizes -> 2."""
    import torch.nn as nn

    act = getattr(nn, activation) if isinstance(activation, str) else activation
    layers, d = [], 2
    for h in hidden_sizes:
        layers += [nn.Linear(d, h), act()]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        d = h
    layers.append(nn.Linear(d, 2))
    return nn.Sequential(*layers)


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def train_mlp(model, X, Y, lr=5e-4, max_epochs=10000, patience=500, seed=42):
    """Full-batch Adam on the MSE loss with early stopping on the training
    loss; the best-epoch weights are restored at the end."""
    import torch
    import torch.nn as nn

    torch.set_num_threads(1)
    set_seed(seed)
    tx = torch.from_numpy(np.asarray(X, np.float32))
    ty = torch.from_numpy(np.asarray(Y, np.float32))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_loss, best_state, wait = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(tx), ty)
        loss.backward()
        opt.step()
        if loss.item() < best_loss:
            best_loss, best_state, wait = loss.item(), copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_mlp(model, X) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(np.asarray(X, np.float32))).numpy().astype(float)


class Standardiser:
    """Z-scoring fitted on the training rows only."""

    def fit(self, A):
        A = np.asarray(A, float)
        self.mean_, self.std_ = A.mean(axis=0), A.std(axis=0) + 1e-8
        return self

    def transform(self, A):
        return (np.asarray(A, float) - self.mean_) / self.std_

    def inverse(self, A):
        return np.asarray(A, float) * self.std_ + self.mean_


def train_bootstrap_member(Xs, Ys, seed, mlp_kw, lr):
    """One ensemble member: a bootstrap resample (with replacement) of the
    standardised training rows and its own weight initialisation, both
    determined by ``seed``."""
    import torch

    torch.set_num_threads(1)
    boot = np.random.default_rng(seed).integers(0, len(Xs), size=len(Xs))
    set_seed(seed)
    net = build_mlp(**mlp_kw)
    train_mlp(net, Xs[boot], Ys[boot], lr=lr, seed=seed)
    return net, boot


class NNEnsemble:
    """Bootstrap ensemble of MLPs with the Palmer et al. linear calibration
    of the member spread, ``sigma_cal = a * sigma_uc + b``."""

    def __init__(self, members, xs: Standardiser, ys: Standardiser, ab: dict | None = None,
                 mlp_kw: dict | None = None):
        self.members, self.xs, self.ys, self.mlp_kw = members, xs, ys, mlp_kw
        self.ab = ab or {o: (1.0, 0.0) for o in OUTPUTS}

    def predict(self, X):
        """Returns (mu, sigma_uc, sigma_cal), each of shape (n, 2), in physical units."""
        Xs = self.xs.transform(X)
        P = np.stack([self.ys.inverse(predict_mlp(m, Xs)) for m in self.members])
        mu, sd = P.mean(axis=0), P.std(axis=0, ddof=1)
        cal = np.column_stack([self.ab[o][0] * sd[:, j] + self.ab[o][1] for j, o in enumerate(OUTPUTS)])
        return mu, sd, cal

    def save(self, directory: str) -> None:
        import json
        import torch
        os.makedirs(directory, exist_ok=True)
        for m, net in enumerate(self.members):
            torch.save(net.state_dict(), os.path.join(directory, f"member_{m:02d}.pt"))
        meta = {"n_members": len(self.members), "mlp_kw": self.mlp_kw,
                "x_mean": self.xs.mean_.tolist(), "x_std": self.xs.std_.tolist(),
                "y_mean": self.ys.mean_.tolist(), "y_std": self.ys.std_.tolist(),
                "calibration": {o: {"a": float(a), "b": float(b)} for o, (a, b) in self.ab.items()}}
        with open(os.path.join(directory, "ensemble.json"), "w") as fh:
            json.dump(meta, fh, indent=1)

    @classmethod
    def load(cls, directory: str, mlp_kw: dict | None = None) -> "NNEnsemble":
        import json
        import torch
        meta = json.load(open(os.path.join(directory, "ensemble.json")))
        mlp_kw = mlp_kw or meta["mlp_kw"]
        members = []
        for m in range(meta["n_members"]):
            net = build_mlp(**mlp_kw)
            net.load_state_dict(torch.load(os.path.join(directory, f"member_{m:02d}.pt"), map_location="cpu"))
            net.eval()
            members.append(net)
        xs, ys = Standardiser(), Standardiser()
        xs.mean_, xs.std_ = np.array(meta["x_mean"]), np.array(meta["x_std"])
        ys.mean_, ys.std_ = np.array(meta["y_mean"]), np.array(meta["y_std"])
        ab = {o: (v["a"], v["b"]) for o, v in meta["calibration"].items()}
        return cls(members, xs, ys, ab, mlp_kw)


# --------------------------------------------------------------------------
# Uncertainty calibration (Palmer et al., npj Comput. Mater. 2022)
# --------------------------------------------------------------------------


def fit_calibration(sigma_uc, resid):
    """Calibration factors (a, b) minimising the Gaussian negative
    log-likelihood of the residuals with standard deviation a*sigma_uc + b."""
    from scipy.optimize import minimize

    s, r = np.asarray(sigma_uc, float), np.asarray(resid, float)

    def nll(p):
        a, b = p
        sc = a * s + b
        if np.any(sc <= 0):
            return 1e12
        return float(np.sum(np.log(sc) + 0.5 * r ** 2 / sc ** 2))

    res = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20000})
    return float(res.x[0]), float(res.x[1])


def rve_bins(sigma, resid, n_bins=15, min_bins_low90=5):
    """Bin the residuals by sigma and fit RMS residual vs sigma by
    count-weighted least squares.  Returns (table, slope, intercept, R2)."""
    sigma, resid = np.asarray(sigma, float), np.asarray(resid, float)
    lo, hi, q90 = sigma.min(), sigma.max(), np.quantile(sigma, 0.9)
    nb = n_bins
    while True:
        edges = np.linspace(lo, hi, nb + 1)
        if (edges[:-1] < q90).sum() >= min_bins_low90 or nb > 200:
            break
        nb += 5
    which = np.clip(np.digitize(sigma, edges) - 1, 0, nb - 1)
    rows = []
    for k in range(nb):
        m = which == k
        if m.sum():
            rows.append({"bin": k, "n": int(m.sum()), "sigma_mean": float(sigma[m].mean()),
                         "rms_resid": float(np.sqrt(np.mean(resid[m] ** 2)))})
    tab = pd.DataFrame(rows)
    w = tab.n.to_numpy(float)
    A = np.column_stack([tab.sigma_mean, np.ones(len(tab))])
    W = np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(A * W, tab.rms_resid.to_numpy() * W[:, 0], rcond=None)
    pred = A @ coef
    ss_res = float(np.sum(w * (tab.rms_resid - pred) ** 2))
    ss_tot = float(np.sum(w * (tab.rms_resid - np.average(tab.rms_resid, weights=w)) ** 2))
    return tab, float(coef[0]), float(coef[1]), (1 - ss_res / ss_tot if ss_tot > 0 else np.nan)


def rve_stats(sigma, resid):
    tab, slope, icpt, r2_ = rve_bins(sigma, resid)
    z = np.asarray(resid, float) / np.asarray(sigma, float)
    return {"r_mean": float(z.mean()), "r_std": float(z.std(ddof=1)),
            "RvE_slope": slope, "RvE_intercept": icpt, "RvE_R2": r2_}, tab


# --------------------------------------------------------------------------
# Geometric overlap (IoU) of reconstructed clusters
# --------------------------------------------------------------------------

R_CAP = 0.15875 + 0.08  # capsule radius of a branch [cm]


def unit_polygons(L_B, L_Y, xB, yB, xY, yY, thB, thY, n_repeat=6):
    """Bar and Y capsules of one cluster (six-fold repeated), as two lists of
    shapely polygons so that the units can be drawn in different colours."""
    from shapely.geometry import LineString, MultiLineString
    from shapely import affinity

    dxB, dyB = L_B * np.cos(thB), L_B * np.sin(thB)
    bar = LineString([(xB - dxB, yB - dyB), (xB + dxB, yB + dyB)]).buffer(R_CAP)
    arms = [LineString([(xY, yY), (xY + L_Y * np.cos(thY + d), yY + L_Y * np.sin(thY + d))])
            for d in (0.0, 2 * np.pi / 3, -2 * np.pi / 3)]
    yshape = MultiLineString(arms).buffer(R_CAP)
    bars = [affinity.rotate(bar, 60 * k, origin=(0, 0)) for k in range(n_repeat)]
    ys = [affinity.rotate(yshape, 60 * k, origin=(0, 0)) for k in range(n_repeat)]
    return bars, ys


def cluster_shape(L_B, L_Y, xB, yB, xY, yY, thB, thY, n_repeat=6):
    """The whole cluster footprint as a single shapely geometry."""
    from shapely.ops import unary_union

    bars, ys = unit_polygons(L_B, L_Y, xB, yB, xY, yY, thB, thY, n_repeat)
    return unary_union(bars + ys)


def iou(a, b) -> float:
    return float(a.intersection(b).area / a.union(b).area)


def reconstruction_iou(L_B, L_Y, true_params, pred_params) -> float:
    """IoU between the simulated and the reconstructed cluster.  ``*_params``
    are ``(xB, xY, yY, thetaB, thetaY)``; ``yB`` is 0 by convention."""
    tB, tY, tyY, tthB, tthY = true_params
    pB, pY, pyY, pthB, pthY = pred_params
    T = cluster_shape(L_B, L_Y, tB, 0.0, tY, tyY, tthB, tthY)
    S = cluster_shape(L_B, L_Y, pB, 0.0, pY, pyY, pthB, pthY)
    return iou(S, T)


def draw_cluster(ax, L_B, L_Y, xB, xY, yY, thB, thY, outline=False, alpha=0.5, lw=1.6):
    """Draw one cluster: filled (prediction) or dashed outline (simulation)."""
    from matplotlib.patches import Polygon as MplPolygon

    bars, ys = unit_polygons(L_B, L_Y, xB, 0.0, xY, yY, thB, thY)
    for polys, color, z in ((bars, GREY, 1), (ys, BLUE, 3)):
        for p in polys:
            for g in (p.geoms if p.geom_type == "MultiPolygon" else [p]):
                xy_ = np.asarray(g.exterior.coords)
                if outline:
                    ax.add_patch(MplPolygon(xy_, fc="none", ec=color, ls="--", lw=lw, zorder=z + 10))
                else:
                    ax.add_patch(MplPolygon(xy_, fc=color, ec="none", alpha=alpha, zorder=z))


# --------------------------------------------------------------------------
# Pareto sets
# --------------------------------------------------------------------------


def pareto_front_max(F: np.ndarray) -> np.ndarray:
    """Boolean mask of the non-dominated rows of a 2-D objective matrix, both
    objectives maximised (ties kept)."""
    F = np.asarray(F, float)
    order = np.lexsort((-F[:, 1], -F[:, 0]))
    keep = np.zeros(len(F), dtype=bool)
    best2 = -np.inf
    for i in order:
        if F[i, 1] > best2:
            keep[i] = True
            best2 = F[i, 1]
    return keep


def design_grid(step: float = 0.02):
    """Regular grid over the design box, returned as (GB, GY, Xgrid)."""
    gb = np.arange(L_BAR_BOUNDS[0], L_BAR_BOUNDS[1] + 1e-9, step)
    gy = np.arange(L_Y_BOUNDS[0], L_Y_BOUNDS[1] + 1e-9, step)
    GB, GY = np.meshgrid(gb, gy)
    return GB, GY, np.column_stack([GB.ravel(), GY.ravel()])
