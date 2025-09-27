#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agentic ML Pipeline – Runner 100% executável (offline)
Arquitetura inspirada em A2A/MCP, com 4 papéis:
  - Supervisor        (coordena a conversa e o fluxo)
  - Data Manager      (lista/carrega dataset e faz feature engineering)
  - ML Expert         (treina RandomForest e gera previsões/métricas)
  - Data Visualizer   (gera gráfico pred vs real)
Sem chaves de API. Sem dependências externas além de pandas/numpy/sklearn/matplotlib.
"""

import os
import json
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

ARTIFACTS_DIR = os.path.abspath("./artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# =========================================================
# "MCP" LOCAL (stubs executáveis)
# =========================================================

def mcp_list_datasets() -> List[str]:
    """
    Lista datasets disponíveis no "MCP local".
    """
    return [
        "diabetes",              # sklearn - regressão (10 features)
        "california_housing",    # sklearn - regressão (8 features)
        "synthetic_regression"   # make_regression - gerado aleatoriamente
    ]


def mcp_load_dataset(name: str) -> Dict[str, Any]:
    """
    Carrega dataset do MCP local e devolve um dicionário com:
      {"X": np.ndarray (n_samples, n_features),
       "y": np.ndarray (n_samples,),
       "feature_names": List[str],
       "target_name": str}
    """
    name = name.lower().strip()
    if name == "diabetes":
        ds = load_diabetes()
        X, y = ds.data, ds.target
        feature_names = [str(f) for f in ds.feature_names]
        target_name = "disease_progression"
    elif name == "california_housing":
        ds = fetch_california_housing()
        X, y = ds.data, ds.target
        feature_names = [str(f) for f in ds.feature_names]
        target_name = "median_house_value"
    elif name == "synthetic_regression":
        X, y = make_regression(
            n_samples=1200,
            n_features=12,
            n_informative=10,
            noise=12.5,
            random_state=42
        )
        feature_names = [f"f{i:02d}" for i in range(X.shape[1])]
        target_name = "target"
    else:
        raise ValueError(f"Dataset '{name}' não encontrado. Use um de: {mcp_list_datasets()}")
    return {"X": X, "y": y, "feature_names": feature_names, "target_name": target_name}


def py_feature_engineer(X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Aplica transformações simples:
      - imputação (se houver NaNs) com média
      - scaling (StandardScaler)
    Retorna X_transformado e metadados do pipeline.
    """
    X_proc = X.copy()
    # imputação simples
    if np.isnan(X_proc).any():
        col_means = np.nanmean(X_proc, axis=0)
        inds = np.where(np.isnan(X_proc))
        X_proc[inds] = np.take(col_means, inds[1])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_proc)

    meta = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "transformations": ["imputation(mean)", "standard_scaler"]
    }
    return X_scaled, meta


def py_train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 400,
    max_depth: int = None,
) -> Dict[str, Any]:
    """
    Treina RandomForestRegressor e retorna métricas e previsões.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    return {
        "model": model,
        "y_true": y_test,
        "y_pred": y_pred,
        "metrics": {"r2": r2, "rmse": rmse},
        "test_indices": None  # não precisamos neste exemplo
    }


def py_plot_scatter(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Gera um scatter plot (y_true vs y_pred) e salva em artifacts/.
    Retorna o caminho do arquivo.
    """
    fig = plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    # linha ideal
    minv = min(np.min(y_true), np.min(y_pred))
    maxv = max(np.max(y_true), np.max(y_pred))
    plt.plot([minv, maxv], [minv, maxv], linestyle="--")
    plt.xlabel("Real (y_true)")
    plt.ylabel("Previsto (y_pred)")
    plt.title("Pred vs Real (RandomForest)")
    plt.tight_layout()

    out_path = os.path.join(ARTIFACTS_DIR, "pred_vs_real.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path

# =========================================================
# Infra “Agent-like” simples (sem LLM) para manter o fluxo
# =========================================================

@dataclass
class Context:
    user_intent: str
    shared: Dict[str, Any] = field(default_factory=dict)


class Supervisor:
    def __init__(self, prefer_dataset: str = None):
        self.prefer_dataset = prefer_dataset

    def plan(self, ctx: Context) -> Dict[str, Any]:
        # Escolhe dataset (preferência -> diabetes -> california -> synthetic)
        available = mcp_list_datasets()
        ds = (self.prefer_dataset or "").lower()
        if ds not in [d.lower() for d in available]:
            # heurística simples
            if "casa" in ctx.user_intent.lower() or "imóvel" in ctx.user_intent.lower():
                chosen = "california_housing"
            elif "diabetes" in ctx.user_intent.lower():
                chosen = "diabetes"
            else:
                chosen = "diabetes"  # default bom para regressão
        else:
            chosen = ds

        plan = {
            "dataset": chosen,
            "target": "auto",  # deixamos como automático
            "visual_to_generate": ["pred_vs_real_scatter"]
        }
        return plan

    def handoff_to_data_manager(self, ctx: Context, plan: Dict[str, Any]):
        ctx.shared["plan"] = plan


class DataManager:
    def run(self, ctx: Context):
        plan = ctx.shared["plan"]
        dataset_name = plan["dataset"]
        ds = mcp_load_dataset(dataset_name)
        X, y = ds["X"], ds["y"]
        feat_names = ds["feature_names"]

        X_fe, meta = py_feature_engineer(X)

        ctx.shared.update({
            "X": X_fe,
            "y": y,
            "feature_names": feat_names,
            "fe_meta": meta,
            "dataset_loaded": dataset_name
        })


class MLExpert:
    def run(self, ctx: Context):
        X, y = ctx.shared["X"], ctx.shared["y"]
        results = py_train_random_forest(X, y)

        ctx.shared.update({
            "model": results["model"],
            "y_true": results["y_true"],
            "y_pred": results["y_pred"],
            "metrics": results["metrics"]
        })


class DataVisualizer:
    def run(self, ctx: Context):
        y_true, y_pred = ctx.shared["y_true"], ctx.shared["y_pred"]
        plot_path = py_plot_scatter(y_true, y_pred)
        ctx.shared["plot_path"] = plot_path


# =========================================================
# Runner (orquestra o fluxo fim-a-fim)
# =========================================================

def run_pipeline(user_task: str, prefer_dataset: str = None) -> Dict[str, Any]:
    ctx = Context(user_intent=user_task)

    supervisor = Supervisor(prefer_dataset=prefer_dataset)
    data_manager = DataManager()
    ml_expert = MLExpert()
    visualizer = DataVisualizer()

    # Supervisor planeja
    plan = supervisor.plan(ctx)
    supervisor.handoff_to_data_manager(ctx, plan)

    # Data Manager prepara dados
    data_manager.run(ctx)

    # ML Expert treina e avalia
    ml_expert.run(ctx)

    # Visualizer plota
    visualizer.run(ctx)

    # Supervisor consolida resposta
    response = {
        "dataset": ctx.shared["dataset_loaded"],
        "metrics": ctx.shared["metrics"],
        "plot_path": ctx.shared["plot_path"],
        "feature_engineering": ctx.shared["fe_meta"]["transformations"],
    }
    return response


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Agentic ML Pipeline Runner")
    parser.add_argument(
        "--task",
        type=str,
        required=False,
        default="Treinar um modelo de regressão e ver o scatter plot.",
        help="Objetivo do usuário (texto livre)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default=None,
        choices=[None, "diabetes", "california_housing", "synthetic_regression"],
        help="(Opcional) Força a escolha de dataset."
    )
    args = parser.parse_args()

    result = run_pipeline(args.task, prefer_dataset=args.dataset)
    print("\n=== RESULTADO FINAL ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\nImagem gerada:", result["plot_path"])
    print("Pronto! ✅")

if __name__ == "__main__":
    main()
