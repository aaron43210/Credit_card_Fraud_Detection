"""
Professional Streamlit app for PROJECT package.
Models in scope: Decision Tree, XGBoost, HGNN.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Ensure local src imports work from PROJECT folder.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DT_MODEL_PATH,
    XGB_MODEL_PATH,
    HGNN_ATT_TD_PATH,
    SCALER_PATH,
    FEATURE_NAMES_PATH,
    LABEL_ENCODERS_PATH,
)  # noqa: E402

# Avoid aggressive OpenMP thread fan-out and reduce libomp instability.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

st.set_page_config(
    page_title="PROJECT Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .app-header {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        background: linear-gradient(
            120deg,
            #0f172a 0%,
            #1f2937 50%,
            #111827 100%
        );
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    .app-header h1 {
        margin: 0;
        color: #f8fafc;
        font-size: 1.85rem;
    }
    .app-header p {
        margin: 0.4rem 0 0 0;
        color: #cbd5e1;
    }
    .kpi {
        padding: 0.9rem;
        border: 1px solid #334155;
        border-radius: 10px;
        background: #0b1220;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _exists(path: Path) -> bool:
    return path.exists() and path.is_file()


@st.cache_resource
def load_model(model_name: str):
    if model_name == "Decision Tree":
        return joblib.load(DT_MODEL_PATH) if _exists(DT_MODEL_PATH) else None

    if model_name == "XGBoost":
        model = joblib.load(XGB_MODEL_PATH) if _exists(XGB_MODEL_PATH) else None
        if model is not None:
            try:
                model.set_params(n_jobs=1)
            except Exception:
                pass
        return model

    if model_name == "HGNN Attention TD":
        if not _exists(HGNN_ATT_TD_PATH):
            return None
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(
            HGNN_ATT_TD_PATH,
            map_location=device,
            weights_only=False,
        )
        return {"checkpoint": checkpoint, "device": device}

    return None


def get_available_models() -> list[str]:
    models = []
    if _exists(DT_MODEL_PATH):
        models.append("Decision Tree")
    if _exists(XGB_MODEL_PATH):
        models.append("XGBoost")
    if _exists(HGNN_ATT_TD_PATH):
        models.append("HGNN Attention TD")
    return models


@st.cache_resource
def load_artifacts() -> dict:
    artifacts = {}

    if _exists(SCALER_PATH):
        artifacts["scaler"] = joblib.load(SCALER_PATH)

    if _exists(FEATURE_NAMES_PATH):
        artifacts["feature_names"] = joblib.load(FEATURE_NAMES_PATH)

    if _exists(LABEL_ENCODERS_PATH):
        artifacts["label_encoders"] = joblib.load(LABEL_ENCODERS_PATH)

    return artifacts


def prepare_tabular_features(
    df_in: pd.DataFrame,
    artifacts: dict,
) -> pd.DataFrame:
    df = df_in.copy()
    feature_names = artifacts.get("feature_names")
    if not feature_names:
        return df

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Apply training-time label encoders when categorical columns are present.
    label_encoders = artifacts.get("label_encoders", {})
    for col, encoder in label_encoders.items():
        if col in df.columns:
            vals = df[col].astype(str)
            class_to_idx = {c: i for i, c in enumerate(encoder.classes_)}
            fallback = class_to_idx.get("Unknown", 0)
            df[col] = vals.map(class_to_idx).fillna(fallback)

    # Ensure scaler/model input is numeric even for arbitrary upload schemas.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(0)

    scaler = artifacts.get("scaler")
    if scaler is not None:
        scaled = scaler.transform(df)
        return pd.DataFrame(scaled, columns=feature_names, index=df.index)

    return df


def build_hgnn_inference_graph(
    df_input: pd.DataFrame,
    artifacts: dict,
):
    """Build the HGNN heterograph for inference without requiring labels."""
    import torch
    from torch_geometric.data import HeteroData

    df = df_input.reset_index(drop=True).copy()
    x_df = prepare_tabular_features(df, artifacts)

    data = HeteroData()
    data["transaction"].x = torch.tensor(x_df.values, dtype=torch.float32)

    tx_indices = np.arange(len(df), dtype=np.int64)

    if "card1" in df.columns:
        card_vals = pd.to_numeric(
            df["card1"],
            errors="coerce",
        ).fillna(-1).values
    else:
        card_vals = np.full(len(df), -1)
    _, card_idx = np.unique(card_vals, return_inverse=True)
    data["card"].x = torch.ones(
        (len(np.unique(card_idx)), 1),
        dtype=torch.float32,
    )
    card_edge_index = torch.tensor([tx_indices, card_idx], dtype=torch.long)
    data["transaction", "used", "card"].edge_index = card_edge_index
    data["card", "used_by", "transaction"].edge_index = card_edge_index.flip(
        [0]
    )

    if "DeviceInfo" in df.columns:
        device_vals = df["DeviceInfo"].astype(str).fillna("Unknown").values
    elif "DeviceType" in df.columns:
        device_vals = df["DeviceType"].astype(str).fillna("Unknown").values
    else:
        device_vals = np.array(["Unknown"] * len(df))
    _, device_idx = np.unique(device_vals, return_inverse=True)
    data["device"].x = torch.ones(
        (len(np.unique(device_idx)), 1),
        dtype=torch.float32,
    )
    device_edge_index = torch.tensor(
        [tx_indices, device_idx],
        dtype=torch.long,
    )
    data["transaction", "on", "device"].edge_index = device_edge_index
    data["device", "hosts", "transaction"].edge_index = device_edge_index.flip(
        [0]
    )

    if "TransactionDT" in df.columns:
        tx_time = pd.to_numeric(
            df["TransactionDT"],
            errors="coerce",
        ).fillna(0).values.astype(np.float32)
    else:
        tx_time = np.arange(len(df), dtype=np.float32)
    age = tx_time.max() - tx_time
    age_scale = np.maximum(np.percentile(age, 95), 1.0)
    decay_np = np.exp(-age / age_scale).astype(np.float32)
    data["transaction"].time_decay = torch.from_numpy(decay_np).unsqueeze(-1)

    return data


def predict_tabular_model(
    df_features: pd.DataFrame,
    model,
    model_name: str,
) -> pd.DataFrame:
    def _align_for_model(df_in: pd.DataFrame, model_obj) -> pd.DataFrame:
        # Prefer exact feature-name alignment when available.
        if hasattr(model_obj, "feature_names_in_"):
            cols = list(model_obj.feature_names_in_)
            return df_in.reindex(columns=cols, fill_value=0)

        expected = int(getattr(model_obj, "n_features_in_", df_in.shape[1]))
        if df_in.shape[1] < expected:
            missing = expected - df_in.shape[1]
            for i in range(missing):
                df_in[f"_pad_{i}"] = 0
        return df_in.iloc[:, :expected]

    out = pd.DataFrame(index=df_features.index)
    aligned = _align_for_model(df_features.copy(), model)

    if model_name == "Decision Tree":
        probs = model.predict_proba(aligned)[:, 1]
    elif model_name == "XGBoost":
        probs = model.predict_proba(aligned)[:, 1]
    else:
        raise ValueError(f"Unsupported tabular model: {model_name}")

    out["fraud_probability"] = probs
    out["prediction"] = (out["fraud_probability"] >= 0.5).astype(int)
    out["risk_label"] = np.where(
        out["prediction"] == 1,
        "HIGH_RISK",
        "LOW_RISK",
    )
    out["model"] = model_name
    return out


def run_hgnn_batch_inference(
    df_input: pd.DataFrame,
    artifacts: dict,
    hgnn_bundle: dict,
) -> pd.DataFrame:
    import torch
    from src.models import build_neural_network

    hetero_data = build_hgnn_inference_graph(df_input, artifacts)

    bundle = hgnn_bundle
    input_dim = int(hetero_data["transaction"].x.shape[1])
    hgnn, _, _ = build_neural_network(
        input_dim=input_dim,
        metadata=hetero_data.metadata(),
    )
    hgnn.load_state_dict(bundle["checkpoint"]["model_state_dict"])
    hgnn = hgnn.to(bundle["device"]).eval()

    hetero_data = hetero_data.to(bundle["device"])
    with torch.no_grad():
        tx_decay = getattr(hetero_data["transaction"], "time_decay", None)
        logits = hgnn(
            hetero_data.x_dict,
            hetero_data.edge_index_dict,
            tx_time_decay=tx_decay,
        )
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    result = df_input.reset_index(drop=True).copy()
    result["fraud_probability"] = probs
    result["prediction"] = (result["fraud_probability"] >= 0.5).astype(int)
    result["risk_label"] = np.where(
        result["prediction"] == 1,
        "HIGH_RISK",
        "LOW_RISK",
    )
    result["model"] = "HGNN"
    return result


artifacts = load_artifacts()
available_models = get_available_models()

st.markdown(
    """
    <div class="app-header">
      <h1>PROJECT Fraud Detection</h1>
      <p>Choose model and score each uploaded transaction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, mid, right = st.columns(3)
with left:
    status_text = "Available" if _exists(DT_MODEL_PATH) else "Missing"
    st.markdown(
        '<div class="kpi"><b>Decision Tree</b><br>' + status_text + "</div>",
        unsafe_allow_html=True,
    )
with mid:
    status_text = "Available" if _exists(XGB_MODEL_PATH) else "Missing"
    st.markdown(
        '<div class="kpi"><b>XGBoost</b><br>' + status_text + "</div>",
        unsafe_allow_html=True,
    )
with right:
    active_device = "Available" if _exists(HGNN_MODEL_PATH) else "Missing"
    st.markdown(
        '<div class="kpi"><b>HGNN</b><br>'
        + active_device
        + "</div>",
        unsafe_allow_html=True,
    )

st.write("")
page = st.sidebar.radio(
    "Navigation",
    ["Batch Prediction", "Project Status"],
    index=0,
)

if page == "Batch Prediction":
    st.subheader("CSV Batch Prediction")
    st.caption("Select a model and run predictions for each transaction row.")

    if not available_models:
        st.error("No model artifacts found in the models folder.")
        st.stop()

    selected_model = st.selectbox("Select model", available_models, index=0)
    if selected_model == "HGNN":
        st.info(
            "HGNN now supports unlabeled CSV inference in uploaded row order."
        )

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:
        df_raw = pd.read_csv(file)
        st.write("Preview", df_raw.head(5))

        try:
            with st.spinner(f"Loading {selected_model}..."):
                selected_model_obj = load_model(selected_model)
            if selected_model_obj is None:
                st.error(f"{selected_model} artifact is missing or failed to load.")
                st.stop()

            if selected_model == "HGNN":
                with st.spinner("Running HGNN inference..."):
                    result = run_hgnn_batch_inference(
                        df_raw,
                        artifacts,
                        selected_model_obj,
                    )
            else:
                with st.spinner(f"Running {selected_model} inference..."):
                    df_features = prepare_tabular_features(df_raw, artifacts)
                    pred = predict_tabular_model(
                        df_features,
                        selected_model_obj,
                        selected_model,
                    )
                    result = pd.concat(
                        [
                            df_raw.reset_index(drop=True),
                            pred.reset_index(drop=True),
                        ],
                        axis=1,
                    )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Rows Scored", len(result))
            with c2:
                pred_fraud = int(result["prediction"].sum())
                st.metric("Predicted Fraud", pred_fraud)
            with c3:
                fraud_rate = result["prediction"].mean() * 100
                st.metric("Fraud Rate", f"{fraud_rate:.2f}%")

            st.subheader("Per-Transaction Predictions")
            st.dataframe(result, use_container_width=True, height=520)

            out_file = (
                "project_predictions_"
                f"{selected_model.lower().replace(' ', '_')}.csv"
            )
            st.download_button(
                "Download Predictions CSV",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name=out_file,
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

if page == "Project Status":
    st.subheader("Required Artifacts")

    table = pd.DataFrame(
        [
            ["Decision Tree", str(DT_MODEL_PATH), _exists(DT_MODEL_PATH)],
            ["XGBoost", str(XGB_MODEL_PATH), _exists(XGB_MODEL_PATH)],
            ["HGNN", str(HGNN_MODEL_PATH), _exists(HGNN_MODEL_PATH)],
            ["Scaler", str(SCALER_PATH), _exists(SCALER_PATH)],
            [
                "Feature Names",
                str(FEATURE_NAMES_PATH),
                _exists(FEATURE_NAMES_PATH),
            ],
            [
                "Label Encoders",
                str(LABEL_ENCODERS_PATH),
                _exists(LABEL_ENCODERS_PATH),
            ],
        ],
        columns=["Artifact", "Path", "Available"],
    )

    st.dataframe(table, use_container_width=True)

    st.info(
        "Batch prediction supports Decision Tree, XGBoost, and HGNN. "
        "HGNN supports unlabeled batch inference in uploaded row order."
    )
