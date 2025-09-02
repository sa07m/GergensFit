# Ortho Appliance Fit/Success Predictor

End-to-end ML pipeline that predicts orthodontic appliance **fit risk** and **success probability** using **3D scans (STL)** + **design parameters**.

- STL ➜ voxel grid via `trimesh`
- Late-fusion model: **3D CNN (scan)** + **MLP (tabular)**
- Train / evaluate / single-scan inference CLIs
- Caches voxelized scans for speed
- Ships with a **synthetic data generator** so you can run it immediately

---

## Folder layout

The project creates these folders as you run it:
project/
├─ ortho_fit_project.py
├─ requirements.txt
├─ README.md
├─ data/
│ ├─ scans/ # .stl files (synthetic or your own)
│ └─ labels.csv # parameters + targets
├─ cache/
│ └─ voxels/ # cached voxel numpy grids (auto)
└─ artifacts/
├─ best_model.pt # trained PyTorch model
├─ tabular_scaler.joblib # sklearn scaler for tabular
├─ tabular_schema.json # schema for features
└─ run_config.json # training config + history

