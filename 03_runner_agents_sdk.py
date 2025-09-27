#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agentic ML Pipeline – OpenAI Agents SDK (versão estável e determinística)

- Usa 1 agente (Supervisor) + 1 tool única (pipeline_run) que executa fim-a-fim:
  carregar dataset -> feature engineering -> treinar RandomForest -> gerar gráfico.
- Garante geração de imagem SEM depender da “criatividade” do LLM.
- Continua usando a Agents SDK (o agente chama a tool via tool-calling).
"""

from __future__ import annotations
import os
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# ===== OpenAI Agents SDK =====
from agents import Agent, Runner, function_tool

# Tracing mais leve (evita payloads enormes nos logs)
os.environ.setdefault("AGENTS_TRACING_CAPTURE_TOOL_OUTPUT", "false")
os.environ.setdefault("AGENTS_TRACING_REDACT_TOOL_OUTPUT", "true")

ARTIFACTS_DIR = os.path.abspath("./artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ---------------------------------------------------------
# Utilidades de dados
# ---------------------------------------------------------
def list_datasets() -> List[str]:
    return ["diabetes", "california_housing", "synthetic_regression"]

def load_dataset_np(name: str) -> Dict[str, Any]:
    name = (name or "").lower().strip()
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
            n_samples=1200, n_features=12, n_informative=10,
            noise=12.5, random_state=42
        )
        feature_names = [f"f{i:02d}" for i in range(X.shape[1])]
        target_name = "target"
    else:
        raise ValueError(f"Dataset '{name}' inválido. Opções: {list_datasets()}")
    return {
        "X": np.asarray(X, float),
        "y": np.asarray(y, float),
        "feature_names": feature_names,
        "target_name": target_name,
        "dataset_name": name,
    }

def feature_engineer(X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    Xp = np.asarray(X, float).copy()
    if np.isnan(Xp).any():
        col_means = np.nanmean(Xp, axis=0)
        inds = np.where(np.isnan(Xp))
        Xp[inds] = np.take(col_means, inds[1])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xp)
    meta = {"transformations": ["imputation(mean)", "standard_scaler"]}
    return Xs, meta

def train_rf(
    X: np.ndarray, y: np.ndarray,
    test_size: float = 0.2, random_state: int = 42,
    n_estimators: int = 400, max_depth: Optional[int] = None
) -> Dict[str, Any]:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=random_state, n_jobs=-1
    )
    model.fit(Xtr, ytr)
    y_pred = model.predict(Xte)
    metrics = {
        "r2": float(r2_score(yte, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(yte, y_pred)))
    }
    return {"y_true": yte, "y_pred": y_pred, "metrics": metrics}

def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, out_dir: str = ARTIFACTS_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    mn, mx = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("Real (y_true)")
    plt.ylabel("Previsto (y_pred)")
    plt.title("Pred vs Real (RandomForest)")
    plt.tight_layout()
    out = os.path.join(out_dir, "pred_vs_real.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out

# ---------------------------------------------------------
# Tool única: pipeline_run (garante imagem + métricas)
# ---------------------------------------------------------
@function_tool
def pipeline_run(
    dataset: str = "diabetes",
    test_size: float = 0.2,
    n_estimators: int = 400,
    max_depth: Optional[int] = None
) -> Dict[str, Any]:
    """
    Executa o pipeline completo e SEMPRE grava a imagem:
    - carrega dataset
    - feature engineering
    - treina RandomForest
    - gera gráfico pred vs real (salvo em ./artifacts/pred_vs_real.png)

    Retorna: {"dataset": str, "metrics": {"r2": float, "rmse": float}, "plot_path": str}
    """
    data = load_dataset_np(dataset)
    Xs, _ = feature_engineer(data["X"])
    res = train_rf(Xs, data["y"], test_size=test_size, n_estimators=n_estimators, max_depth=max_depth)
    plot_path = plot_scatter(res["y_true"], res["y_pred"])
    return {"dataset": data["dataset_name"], "metrics": res["metrics"], "plot_path": plot_path}

# ---------------------------------------------------------
# Agente (Supervisor) – chama APENAS a tool pipeline_run
# ---------------------------------------------------------
supervisor = Agent(
    name="Supervisor",
    model="gpt-4o-mini",
    instructions=(
        "Você tem uma única ferramenta chamada pipeline_run que executa todo o pipeline "
        "(carregar dataset -> feature engineering -> treinar RandomForest -> plotar). "
        "Chame pipeline_run UMA vez usando os parâmetros sugeridos pelo usuário. "
        "No final, responda SOMENTE com a última linha no formato:\n"
        "JSON: {\"dataset\":\"<nome>\", \"metrics\":{\"r2\": <num>, \"rmse\": <num>}, \"plot_path\":\"<path>\"}"
    ),
    tools=[pipeline_run],
)

# ---------------------------------------------------------
# CLI e execução
# ---------------------------------------------------------
DEFAULT_TASK = (
    "Quero treinar um modelo de regressão, ver R² e RMSE e gerar o gráfico pred vs real."
)

def main():
    parser = argparse.ArgumentParser(description="Agentic ML Pipeline (Agents SDK, determinístico)")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--dataset", type=str, default="diabetes", choices=list_datasets())
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--n_estimators", type=int, default=400)
    parser.add_argument("--max_depth", type=int, default=0, help="0 = None")
    parser.add_argument("--max_turns", type=int, default=3, help="Só 1 chamada de tool + resposta.")
    args = parser.parse_args()

    md = None if args.max_depth == 0 else args.max_depth
    prompt = (
        f"{args.task}\n"
        f"Parâmetros sugeridos: dataset={args.dataset}, test_size={args.test_size}, "
        f"n_estimators={args.n_estimators}, max_depth={md}.\n"
        "Use pipeline_run exatamente uma vez."
    )

    # Executa de forma síncrona (sem loop longo)
    result = Runner.run_sync(supervisor, prompt, max_turns=args.max_turns)

    print("\n=== SAÍDA FINAL ===")
    print(result.final_output)
    # ajuda extra: extrai path caso o LLM siga o formato
    try:
        line = result.final_output.strip().splitlines()[-1]
        if line.startswith("JSON:"):
            payload = json.loads(line.replace("JSON:", "").strip())
            print("\nImagem gerada em:", payload.get("plot_path"))
    except Exception:
        pass

    print("\nPronto! ✅  Veja o arquivo em ./artifacts/pred_vs_real.png")

if __name__ == "__main__":
    main()
