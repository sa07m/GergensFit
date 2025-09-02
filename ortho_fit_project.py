#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ortho Appliance Fit/Success Predictor
-------------------------------------
End-to-end project:
- Synthetic data generation (STLs + labels.csv)
- 3D voxelization of meshes (using trimesh) into fixed grids
- Late-fusion model (3D CNN + MLP for design parameters)
- Train / Evaluate / Predict CLI

Folder layout (auto-created):
  data/
    scans/                # .stl files
    labels.csv            # id, tabular features, targets
  cache/
    voxels/*.npy          # cached voxel grids
  artifacts/
    best_model.pt
    tabular_scaler.joblib
    tabular_schema.json
    run_config.json

Run:
  python ortho_fit_project.py make_synthetic --n 400 --out_dir data
  python ortho_fit_project.py train --data_dir data --epochs 10
  python ortho_fit_project.py evaluate --data_dir data
  python ortho_fit_project.py predict --scan data/scans/sample_000001.stl --params "thickness=1.2,base_width=31.5,relief_mm=0.2,material_modulus=1.8,offset_mm=0.35,undercut_depth=0.6,arch_form=medium"

Dependencies:
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  pip install numpy pandas scikit-learn joblib tqdm trimesh shapely==1.8.5.post1 networkx
"""
import os
import json
import math
import sys
import time
import random
import argparse
import pathlib
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import dump, load

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import trimesh

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------
# Utilities
# ---------------------------

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def ensure_dir(p: str):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def set_deterministic():
    torch.use_deterministic_algorithms(False)  # safe default for CPU
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# Synthetic Data Generation
# ---------------------------

ARCH_FORMS = ["narrow", "medium", "broad"]

def _random_primitive_points(n_points=2000) -> np.ndarray:
    """Create a random blobby point cloud by sampling union of simple primitives then convex-hull them."""
    clouds = []
    # create a handful of primitives in random positions
    for _ in range(np.random.randint(2, 5)):
        kind = random.choice(["box", "sphere", "cylinder"])
        if kind == "box":
            extents = np.random.uniform(5, 20, size=3)
            box = trimesh.creation.box(extents=extents)
            pts, _ = trimesh.sample.sample_surface(box, n_points//4)
        elif kind == "sphere":
            radius = np.random.uniform(5, 12)
            sph = trimesh.creation.icosphere(subdivisions=2, radius=radius)
            pts, _ = trimesh.sample.sample_surface(sph, n_points//4)
        else:
            height = np.random.uniform(5, 25)
            radius = np.random.uniform(3, 10)
            cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=24)
            pts, _ = trimesh.sample.sample_surface(cyl, n_points//4)
        # random position/orientation
        R = trimesh.transformations.random_rotation_matrix()
        T = trimesh.transformations.translation_matrix(np.random.uniform(-15, 15, size=3))
        M = trimesh.transformations.concatenate_matrices(T, R)
        pts = trimesh.transformations.transform_points(pts, M)
        clouds.append(pts)
    P = np.vstack(clouds)
    return P

def _hull_mesh_from_points(P: np.ndarray) -> trimesh.Trimesh:
    hull = trimesh.Trimesh(vertices=P).convex_hull
    # Smooth-ish by subdividing a bit:
    try:
        hull = hull.subdivide()
    except Exception:
        pass
    hull.remove_degenerate_faces()
    hull.remove_unreferenced_vertices()
    hull.fix_normals()
    return hull

def _make_label_from_params(params: Dict[str, Any]) -> Tuple[int, int]:
    """
    Heuristic for synthetic labels:
    - y_fit_issue: 1 if we predict a fit problem; 0 otherwise
    - y_success: 1 if overall success likely; 0 otherwise
    """
    t = float(params["thickness"])
    w = float(params["base_width"])
    r = float(params["relief_mm"])
    E = float(params["material_modulus"])
    off = float(params["offset_mm"])
    u = float(params["undercut_depth"])
    arch = params["arch_form"]

    # Simple interpretable rule-of-thumb logic:
    # Too little offset relative to thickness + undercut -> fit issues
    fit_risk_score = (t - off) + 0.5 * u - 0.3 * r + 0.02 * max(0, 35 - w)
    if arch == "narrow":
        fit_risk_score += 0.15
    elif arch == "broad":
        fit_risk_score -= 0.05

    y_fit_issue = int(fit_risk_score > 0.4)

    # Success depends inversely on fit risk and positively on stiffness & relief
    success_score = (0.8 - fit_risk_score) + 0.2 * (E / 3.0) + 0.2 * r - 0.05 * abs(w - 32.0)
    y_success = int(success_score > 0.55)

    # Make them not perfectly correlated:
    if y_fit_issue == 1 and y_success == 1 and random.random() < 0.2:
        y_success = 0
    return y_fit_issue, y_success

def generate_synthetic_dataset(n: int, out_dir: str):
    scans_dir = os.path.join(out_dir, "scans")
    ensure_dir(scans_dir)
    rows = []
    print(f"[make_synthetic] Generating {n} synthetic meshes and labels …")
    for i in tqdm(range(n)):
        # random params within plausible ranges
        params = {
            "thickness": round(np.random.uniform(0.6, 2.0), 3),
            "base_width": round(np.random.uniform(26.0, 40.0), 3),
            "relief_mm": round(np.random.uniform(0.0, 0.7), 3),
            "material_modulus": round(np.random.uniform(1.0, 3.2), 3),  # GPa (normalized-ish)
            "offset_mm": round(np.random.uniform(0.1, 0.8), 3),
            "undercut_depth": round(np.random.uniform(0.0, 1.2), 3),
            "arch_form": random.choice(ARCH_FORMS),
        }
        y_fit_issue, y_success = _make_label_from_params(params)

        # create a random "jawish" hull mesh and save STL
        pts = _random_primitive_points(n_points=3000)
        mesh = _hull_mesh_from_points(pts)
        sample_id = f"sample_{i:06d}"
        stl_path = os.path.join(scans_dir, f"{sample_id}.stl")
        mesh.export(stl_path)

        rows.append({
            "id": sample_id,
            **params,
            "y_fit_issue": y_fit_issue,
            "y_success": y_success
        })
    labels = pd.DataFrame(rows)
    labels.to_csv(os.path.join(out_dir, "labels.csv"), index=False)
    print(f"[make_synthetic] Wrote {len(labels)} rows to {os.path.join(out_dir, 'labels.csv')}")


# ---------------------------
# Voxelization
# ---------------------------

def load_mesh_normalized(stl_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(stl_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        # if it's a scene, merge into a single mesh
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate([g for g in mesh.dump()])
        else:
            raise ValueError(f"Unsupported mesh container for {stl_path}")
    mesh.remove_unreferenced_vertices()
    # center and scale to fit in unit cube [-0.5, 0.5]^3
    centroid = mesh.centroid
    mesh.apply_translation(-centroid)
    extents = mesh.extents
    scale = 1.0 / max(extents.max(), 1e-6)
    mesh.apply_scale(0.95 * scale)  # slight margin
    return mesh

def mesh_to_voxel(mesh: trimesh.Trimesh, grid: int = 48, points_per_mesh: int = 12000) -> np.ndarray:
    """
    Sample mesh surface points then bin to a fixed 3D occupancy grid.
    Returns: (1, D, H, W) float32 array with {0,1}.
    """
    pts, _ = trimesh.sample.sample_surface(mesh, count=points_per_mesh)
    # map from [-0.5,0.5] to grid indices
    xyz = np.clip((pts + 0.5) * grid, 0, grid - 1 - 1e-6)
    idx = np.floor(xyz).astype(np.int32)
    occ = np.zeros((grid, grid, grid), dtype=np.uint8)
    occ[idx[:, 2], idx[:, 1], idx[:, 0]] = 1  # (z,y,x)
    # optionally thicken slightly with one pass of 6-neighborhood dilation
    occ_pad = np.pad(occ, 1)
    dil = (
        occ_pad[1:-1,1:-1,1:-1] |
        occ_pad[:-2,1:-1,1:-1] | occ_pad[2:,1:-1,1:-1] |
        occ_pad[1:-1,:-2,1:-1] | occ_pad[1:-1,2:,1:-1] |
        occ_pad[1:-1,1:-1,:-2] | occ_pad[1:-1,1:-1,2:]
    ).astype(np.uint8)
    occ = dil
    occ = occ[None, ...].astype(np.float32)
    return occ  # shape (1, D, H, W)

def voxelize_path(stl_path: str, cache_dir: str, grid: int = 48) -> np.ndarray:
    ensure_dir(cache_dir)
    base = os.path.splitext(os.path.basename(stl_path))[0]
    npy_path = os.path.join(cache_dir, f"{base}_g{grid}.npy")
    if os.path.exists(npy_path):
        return np.load(npy_path)
    mesh = load_mesh_normalized(stl_path)
    vox = mesh_to_voxel(mesh, grid=grid)
    np.save(npy_path, vox)
    return vox


# ---------------------------
# Dataset / Preprocessing
# ---------------------------

@dataclass
class TabularSchema:
    numeric_cols: List[str]
    categorical_cols: List[str]
    cats_values: Dict[str, List[str]]  # fixed order
    label_cols: List[str]

def build_schema(df: pd.DataFrame) -> TabularSchema:
    label_cols = ["y_fit_issue", "y_success"]
    ignore = ["id"] + label_cols
    numeric_cols = ["thickness", "base_width", "relief_mm", "material_modulus", "offset_mm", "undercut_depth"]
    categorical_cols = ["arch_form"]
    # fix cat ordering
    cats_values = {"arch_form": sorted(df["arch_form"].astype(str).unique().tolist())}
    return TabularSchema(numeric_cols, categorical_cols, cats_values, label_cols)

def tabular_to_matrix(df: pd.DataFrame, schema: TabularSchema) -> Tuple[np.ndarray, List[str]]:
    X_num = df[schema.numeric_cols].astype(float).values
    # one-hot cats
    OH_parts = []
    oh_names = []
    for c in schema.categorical_cols:
        vals = schema.cats_values[c]
        idx_map = {v:i for i,v in enumerate(vals)}
        onehot = np.zeros((len(df), len(vals)), dtype=np.float32)
        for r, v in enumerate(df[c].astype(str).values):
            j = idx_map.get(v, None)
            if j is not None:
                onehot[r, j] = 1.0
        OH_parts.append(onehot)
        oh_names += [f"{c}={v}" for v in vals]
    X_cat = np.concatenate(OH_parts, axis=1) if OH_parts else np.zeros((len(df),0))
    X = np.concatenate([X_num, X_cat], axis=1).astype(np.float32)
    feature_names = schema.numeric_cols + oh_names
    return X, feature_names


class OrthoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, scans_dir: str, cache_dir: str, schema: TabularSchema,
                 grid: int = 48, scaler=None):
        self.df = df.reset_index(drop=True)
        self.scans_dir = scans_dir
        self.cache_dir = cache_dir
        self.schema = schema
        self.grid = grid
        self.scaler = scaler

        X, self.feature_names = tabular_to_matrix(self.df, schema)
        self.X_tab = X
        if self.scaler is not None:
            self.X_tab = self.scaler.transform(self.X_tab)
        self.y = self.df[self.schema.label_cols].astype(np.float32).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rid = self.df.loc[idx, "id"]
        stl_path = os.path.join(self.scans_dir, f"{rid}.stl")
        vox = voxelize_path(stl_path, cache_dir=self.cache_dir, grid=self.grid)  # (1,D,H,W)
        vox = torch.from_numpy(vox)
        tab = torch.from_numpy(self.X_tab[idx])
        y = torch.from_numpy(self.y[idx])
        return vox, tab, y, rid


# ---------------------------
# Model: 3D CNN + MLP Fusion
# ---------------------------

class CNN3D(nn.Module):
    def __init__(self, in_ch=1, base=16):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, base, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(base)
        self.conv2 = nn.Conv3d(base, base*2, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(base*2)
        self.conv3 = nn.Conv3d(base*2, base*4, 3, padding=1)
        self.bn3 = nn.BatchNorm3d(base*4)
        self.pool = nn.MaxPool3d(2)
        self.gap = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.gap(x).flatten(1)
        return x  # (B, base*4)

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden=64, out_dim=64, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, x): return self.net(x)

class FusionNet(nn.Module):
    def __init__(self, tab_dim: int):
        super().__init__()
        self.cnn = CNN3D(in_ch=1, base=16)
        self.mlp = MLP(in_dim=tab_dim, hidden=64, out_dim=64, p=0.1)
        self.head = nn.Sequential(
            nn.Linear(16*4 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)   # logits for [y_fit_issue, y_success]
        )

    def forward(self, vox, tab):
        f3d = self.cnn(vox)
        ftab = self.mlp(tab)
        fused = torch.cat([f3d, ftab], dim=1)
        logits = self.head(fused)
        return logits


# ---------------------------
# Training / Evaluation
# ---------------------------

def split_train_val(df: pd.DataFrame, val_frac=0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # simple stratify on y_success if available
    if "y_success" in df.columns:
        pos = df[df["y_success"] == 1]
        neg = df[df["y_success"] == 0]
        n_val_pos = max(1, int(len(pos)*val_frac))
        n_val_neg = max(1, int(len(neg)*val_frac))
        val = pd.concat([pos.sample(n=min(n_val_pos, len(pos)), random_state=SEED),
                         neg.sample(n=min(n_val_neg, len(neg)), random_state=SEED)])
        train = df.drop(val.index)
        return train.sample(frac=1.0, random_state=SEED), val.sample(frac=1.0, random_state=SEED)
    else:
        val = df.sample(frac=val_frac, random_state=SEED)
        train = df.drop(val.index)
        return train, val

def train_model(data_dir: str, artifacts_dir: str, cache_dir: str,
                grid: int = 48, batch_size: int = 8, epochs: int = 10, lr: float = 1e-3):

    ensure_dir(artifacts_dir); ensure_dir(cache_dir)
    labels_path = os.path.join(data_dir, "labels.csv")
    scans_dir = os.path.join(data_dir, "scans")
    df = pd.read_csv(labels_path)
    schema = build_schema(df)

    # Tabular matrices + scaler fit on train
    train_df, val_df = split_train_val(df, val_frac=0.2)
    X_train_tab, feat_names = tabular_to_matrix(train_df, schema)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train_tab)

    dump(scaler, os.path.join(artifacts_dir, "tabular_scaler.joblib"))
    with open(os.path.join(artifacts_dir, "tabular_schema.json"), "w") as f:
        json.dump({
            "numeric_cols": schema.numeric_cols,
            "categorical_cols": schema.categorical_cols,
            "cats_values": schema.cats_values,
            "label_cols": schema.label_cols,
            "feature_names": feat_names
        }, f, indent=2)

    # datasets
    train_ds = OrthoDataset(train_df, scans_dir, cache_dir, schema, grid=grid, scaler=scaler)
    val_ds   = OrthoDataset(val_df,   scans_dir, cache_dir, schema, grid=grid, scaler=scaler)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model = FusionNet(tab_dim=train_ds.X_tab.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_path = os.path.join(artifacts_dir, "best_model.pt")
    history = []

    print(f"[train] device={DEVICE}, batches train={len(train_loader)}, val={len(val_loader)}")
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for vox, tab, y, _ in train_loader:
            vox = vox.to(DEVICE)
            tab = tab.to(DEVICE).float()
            y = y.to(DEVICE)
            opt.zero_grad()
            logits = model(vox, tab)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * vox.size(0)
        train_loss = total_loss / len(train_ds)

        # val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vox, tab, y, _ in val_loader:
                vox = vox.to(DEVICE)
                tab = tab.to(DEVICE).float()
                y = y.to(DEVICE)
                logits = model(vox, tab)
                loss = criterion(logits, y)
                val_loss += loss.item() * vox.size(0)
        val_loss /= len(val_ds)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[train] epoch {epoch:03d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(),
                        "tab_dim": train_ds.X_tab.shape[1]},
                       best_path)
            print(f"[train]  -> saved best to {best_path}")

    with open(os.path.join(artifacts_dir, "run_config.json"), "w") as f:
        json.dump({"grid": grid, "batch_size": batch_size, "epochs": epochs, "lr": lr,
                   "history": history}, f, indent=2)

def sigmoid_np(x): return 1 / (1 + np.exp(-x))

def evaluate_model(data_dir: str, artifacts_dir: str, cache_dir: str, grid: int = 48, batch_size: int = 8):
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_fscore_support, confusion_matrix

    labels_path = os.path.join(data_dir, "labels.csv")
    scans_dir = os.path.join(data_dir, "scans")
    df = pd.read_csv(labels_path)
    schema = build_schema(df)

    scaler = load(os.path.join(artifacts_dir, "tabular_scaler.joblib"))
    val_df = df.sample(frac=0.3, random_state=SEED)  # quick eval on holdout slice
    ds = OrthoDataset(val_df, scans_dir, cache_dir, schema, grid=grid, scaler=scaler)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    ckpt = torch.load(os.path.join(artifacts_dir, "best_model.pt"), map_location=DEVICE)
    model = FusionNet(tab_dim=ckpt["tab_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_logits = []
    all_y = []
    all_ids = []
    with torch.no_grad():
        for vox, tab, y, ids in dl:
            vox = vox.to(DEVICE)
            tab = tab.to(DEVICE).float()
            logits = model(vox, tab).cpu().numpy()
            all_logits.append(logits)
            all_y.append(y.numpy())
            all_ids.extend(ids)
    logits = np.vstack(all_logits)
    y_true = np.vstack(all_y)
    y_prob = sigmoid_np(logits)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {}
    for j, name in enumerate(["y_fit_issue", "y_success"]):
        try:
            roc = roc_auc_score(y_true[:, j], y_prob[:, j])
        except Exception:
            roc = float("nan")
        ap = average_precision_score(y_true[:, j], y_prob[:, j])
        f1 = f1_score(y_true[:, j], y_pred[:, j])
        pr, re, f1p, _ = precision_recall_fscore_support(y_true[:, j], y_pred[:, j], average='binary', zero_division=0)
        cm = confusion_matrix(y_true[:, j], y_pred[:, j]).tolist()
        metrics[name] = {"roc_auc": float(roc), "avg_precision": float(ap),
                         "f1": float(f1), "precision": float(pr), "recall": float(re), "confusion_matrix": cm}

    print(json.dumps(metrics, indent=2))
    return metrics

def load_artifacts(artifacts_dir: str):
    scaler = load(os.path.join(artifacts_dir, "tabular_scaler.joblib"))
    with open(os.path.join(artifacts_dir, "tabular_schema.json"), "r") as f:
        schema_json = json.load(f)
    schema = TabularSchema(
        numeric_cols=schema_json["numeric_cols"],
        categorical_cols=schema_json["categorical_cols"],
        cats_values=schema_json["cats_values"],
        label_cols=schema_json["label_cols"],
    )
    ckpt = torch.load(os.path.join(artifacts_dir, "best_model.pt"), map_location=DEVICE)
    model = FusionNet(tab_dim=ckpt["tab_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    # load grid config
    grid = 48
    rc = os.path.join(artifacts_dir, "run_config.json")
    if os.path.exists(rc):
        with open(rc, "r") as f:
            grid = json.load(f).get("grid", 48)
    return model, scaler, schema, grid

def parse_params_str(params_str: str, schema: TabularSchema) -> Dict[str, Any]:
    # "k=v,k=v,arch_form=medium"
    raw = {}
    for piece in params_str.split(","):
        if "=" not in piece: continue
        k,v = piece.split("=",1)
        raw[k.strip()] = v.strip()
    # coerce and fill defaults if missing
    defaults = {
        "thickness": 1.0, "base_width": 32.0, "relief_mm": 0.2,
        "material_modulus": 2.0, "offset_mm": 0.4, "undercut_depth": 0.5,
        "arch_form": "medium"
    }
    for k in defaults:
        if k not in raw:
            raw[k] = defaults[k]
    # cast numerics
    for k in schema.numeric_cols:
        raw[k] = float(raw[k])
    raw["arch_form"] = str(raw["arch_form"])
    return raw

def single_inference(model, scaler, schema, grid: int, scan_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    # voxelize
    vox = voxelize_path(scan_path, cache_dir=os.path.join("cache", "voxels"), grid=grid)
    vox_t = torch.from_numpy(vox)[None, ...].to(DEVICE)  # add batch

    # tabular vector
    df = pd.DataFrame([{**params}])
    X, feat_names = tabular_to_matrix(df.assign(id="X", y_fit_issue=0, y_success=0), schema)
    X = scaler.transform(X)
    tab_t = torch.from_numpy(X).float().to(DEVICE)

    with torch.no_grad():
        logits = model(vox_t, tab_t).cpu().numpy()[0]
        probs = sigmoid_np(logits)
    out = {
        "p_fit_issue": float(probs[0]),
        "p_success": float(probs[1]),
        "flag_fit_risk": bool(probs[0] >= 0.5),
        "flag_success": bool(probs[1] >= 0.5)
    }
    return out


# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Ortho 3D Fit/Success ML")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_syn = sub.add_parser("make_synthetic", help="Generate synthetic dataset (STLs + labels.csv)")
    p_syn.add_argument("--n", type=int, default=400)
    p_syn.add_argument("--out_dir", type=str, default="data")

    p_tr = sub.add_parser("train", help="Train model")
    p_tr.add_argument("--data_dir", type=str, default="data")
    p_tr.add_argument("--artifacts_dir", type=str, default="artifacts")
    p_tr.add_argument("--cache_dir", type=str, default=os.path.join("cache", "voxels"))
    p_tr.add_argument("--grid", type=int, default=48)
    p_tr.add_argument("--batch_size", type=int, default=8)
    p_tr.add_argument("--epochs", type=int, default=10)
    p_tr.add_argument("--lr", type=float, default=1e-3)

    p_ev = sub.add_parser("evaluate", help="Evaluate model")
    p_ev.add_argument("--data_dir", type=str, default="data")
    p_ev.add_argument("--artifacts_dir", type=str, default="artifacts")
    p_ev.add_argument("--cache_dir", type=str, default=os.path.join("cache", "voxels"))
    p_ev.add_argument("--grid", type=int, default=48)
    p_ev.add_argument("--batch_size", type=int, default=8)

    p_pr = sub.add_parser("predict", help="Predict on a single scan + params")
    p_pr.add_argument("--scan", type=str, required=True, help="Path to .stl")
    p_pr.add_argument("--params", type=str, required=True,
                      help="Comma-separated k=v list. "
                           "Example: thickness=1.2,base_width=31.5,relief_mm=0.2,material_modulus=1.8,offset_mm=0.35,undercut_depth=0.6,arch_form=medium")
    p_pr.add_argument("--artifacts_dir", type=str, default="artifacts")

    args = parser.parse_args()
    set_deterministic()

    if args.cmd == "make_synthetic":
        generate_synthetic_dataset(n=args.n, out_dir=args.out_dir)

    elif args.cmd == "train":
        train_model(
            data_dir=args.data_dir,
            artifacts_dir=args.artifacts_dir,
            cache_dir=args.cache_dir,
            grid=args.grid,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr
        )

    elif args.cmd == "evaluate":
        evaluate_model(
            data_dir=args.data_dir,
            artifacts_dir=args.artifacts_dir,
            cache_dir=args.cache_dir,
            grid=args.grid,
            batch_size=args.batch_size
        )

    elif args.cmd == "predict":
        model, scaler, schema, grid = load_artifacts(args.artifacts_dir)
        params = parse_params_str(args.params, schema)
        res = single_inference(model, scaler, schema, grid, scan_path=args.scan, params=params)
        print(json.dumps({"scan": args.scan, "params": params, "prediction": res}, indent=2))

if __name__ == "__main__":
    main()
