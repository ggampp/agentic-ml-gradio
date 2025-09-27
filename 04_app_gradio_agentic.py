#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gradio Chatbot – Agentic ML (IDs + tool única, sem recursão)

- UI rica: Chatbot, Dropdown, Upload CSV (placeholder), Sliders, Preview (25 linhas),
  Métricas (JSON), Gráfico (Image), Acordeões, Abas, Exemplos, Botões utilitários.
- Execução segura: 1 agente (Supervisor) chamando UMA tool determinística
  (pipeline_run_ids), que usa Object Store (IDs) e SEMPRE salva a imagem.
- Sem listas gigantes nos prompts: arrays ficam no STORE.
- Fallback OFFLINE quando OPENAI_API_KEY estiver ausente.
"""

from __future__ import annotations
import os, json, uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from gradio.components import ChatMessage

from sklearn.datasets import load_diabetes, fetch_california_housing, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ====== Agents SDK (opcional) ======
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
AGENTS_AVAILABLE = False
try:
    from agents import Agent, Runner, function_tool  # type: ignore
    AGENTS_AVAILABLE = bool(OPENAI_KEY)
except Exception:
    AGENTS_AVAILABLE = False

# Tracing “magro” para não capturar outputs massivos das tools
os.environ.setdefault("AGENTS_TRACING_CAPTURE_TOOL_OUTPUT", "false")
os.environ.setdefault("AGENTS_TRACING_REDACT_TOOL_OUTPUT", "true")

ARTIFACTS_DIR = os.path.abspath("./artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# =========================================================
# Object Store (IDs → objetos em memória)
# =========================================================
STORE: Dict[str, Any] = {}

def put(obj: Any) -> str:
    k = str(uuid.uuid4())
    STORE[k] = obj
    return k

def get(k: str) -> Any:
    if k not in STORE:
        raise KeyError(f"data_id '{k}' não encontrado no STORE.")
    return STORE[k]

# =========================================================
# Dados utilitários
# =========================================================
def list_datasets() -> List[str]:
    # Você pode reintroduzir ames_housing/bikesharing depois; aqui mantemos leves
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

def feature_engineer_np(X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    Xp = np.asarray(X, float).copy()
    if np.isnan(Xp).any():
        col_means = np.nanmean(Xp, axis=0)
        inds = np.where(np.isnan(Xp))
        Xp[inds] = np.take(col_means, inds[1])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xp)
    meta = {"transformations": ["imputation(mean)", "standard_scaler"]}
    return Xs, meta

def train_rf_np(
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
    return {
        "y_true": yte,
        "y_pred": y_pred,
        "metrics": {
            "r2": float(r2_score(yte, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(yte, y_pred)))
        }
    }

def plot_scatter_np(y_true: np.ndarray, y_pred: np.ndarray, out_dir: str = ARTIFACTS_DIR) -> str:
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

# =========================================================
# Tool única (IDs) – usada pelo agente e no modo offline
# =========================================================
def pipeline_run_ids_impl(
    dataset: str = "diabetes",
    test_size: float = 0.2,
    n_estimators: int = 400,
    max_depth: Optional[int] = None
) -> Dict[str, Any]:
    """Executa: load -> FE -> treino -> plot, trafegando somente IDs."""
    data = load_dataset_np(dataset)
    X_id, y_id = put(data["X"]), put(data["y"])

    Xs, _ = feature_engineer_np(get(X_id))
    Xs_id = put(Xs)

    res = train_rf_np(get(Xs_id), get(y_id),
                      test_size=test_size, n_estimators=n_estimators, max_depth=max_depth)
    y_true_id, y_pred_id = put(res["y_true"]), put(res["y_pred"])

    plot_path = plot_scatter_np(get(y_true_id), get(y_pred_id))
    return {"dataset": data["dataset_name"], "metrics": res["metrics"], "plot_path": plot_path}

# ===== Exposição como tool da Agents SDK (se disponível) =====
if AGENTS_AVAILABLE:
    from agents import Agent, Runner, function_tool  # type: ignore

    @function_tool
    def pipeline_run_ids(
        dataset: str = "diabetes",
        test_size: float = 0.2,
        n_estimators: int = 400,
        max_depth: Optional[int] = None
    ) -> Dict[str, Any]:
        return pipeline_run_ids_impl(dataset, test_size, n_estimators, max_depth)

    supervisor = Agent(
        name="Supervisor",
        model="gpt-4o-mini",
        instructions=(
            "Você possui UMA ferramenta chamada pipeline_run_ids que executa todo o pipeline "
            "(carregar dataset -> feature engineering -> treinar RandomForest -> plotar), usando IDs no backend. "
            "Chame pipeline_run_ids uma única vez com os parâmetros fornecidos. "
            "Responda somente com a última linha no formato:\n"
            "JSON: {\"dataset\":\"<nome>\", \"metrics\":{\"r2\": <num>, \"rmse\": <num>}, \"plot_path\":\"<path>\"}"
        ),
        tools=[pipeline_run_ids],
    )

# =========================================================
# FUNÇÕES da UI
# =========================================================
def preview_dataset(dataset: str) -> Tuple[str, pd.DataFrame]:
    d = load_dataset_np(dataset)
    df = pd.DataFrame(d["X"], columns=d["feature_names"])
    df["__target__"] = d["y"]
    return f"Preview de '{dataset}' (25 linhas mostradas)", df.head(25)



async def run_pipeline_ui(
    message: str,
    history: list[gr.ChatMessage],    # <---
    dataset: str,
    test_size: float,
    n_estimators: int,
    max_depth_slider: int
):
    history = history or []
    # adiciona a mensagem do usuário
    history.append(gr.ChatMessage(role="user", content=message))

    max_depth = None if int(max_depth_slider) == 0 else int(max_depth_slider)

    # --- resto do pipeline ---
    payload = pipeline_run_ids_impl(dataset, test_size, n_estimators, max_depth)
    reply = (
        f"[OFFLINE] Execução concluída.\n"
        f"Dataset: {payload['dataset']}\n"
        f"R²={payload['metrics']['r2']:.4f} | RMSE={payload['metrics']['rmse']:.4f}\n"
        f"Plot: {payload['plot_path']}"
    )

    # adiciona a resposta do assistente
    history.append(gr.ChatMessage(role="assistant", content=reply))
    return history, payload["metrics"], payload["plot_path"]


def clear_all():
    return [], {}, None, "Pronto."

# =========================================================
# UI GRADIO
# =========================================================
with gr.Blocks(title="Agentic ML – Chatbot (IDs, sem recursão)", fill_height=True) as demo:
    gr.Markdown(
        """
        # 🤖 Agentic ML – Chatbot (IDs, sem recursão)
        **Pipeline**: Load → Feature Eng. → RandomForest → Plot • **Sem recursão** • **Sem arrays no contexto**
        """
    )
    with gr.Tabs():
        with gr.TabItem("Pipeline"):
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="Conversa", height=360, type="messages")
                    with gr.Row():
                        user_msg = gr.Textbox(
                            label="Mensagem",
                            placeholder="Ex.: Treine e gere o gráfico pred vs real.",
                            autofocus=True
                        )
                        send_btn = gr.Button("Enviar", variant="primary")
                    gr.Examples(
                        [["Quero treinar e ver as métricas."],
                         ["Use o dataset california_housing."],
                         ["Apenas rode o pipeline com as configurações atuais."]],
                        inputs=[user_msg],
                        label="Exemplos"
                    )
                with gr.Column(scale=1):
                    with gr.Accordion("Configurações do Dataset", open=True):
                        dataset_dd = gr.Dropdown(
                            label="Dataset",
                            choices=list_datasets(),
                            value="diabetes"
                        )
                        preview_btn = gr.Button("Pré-visualizar (25 linhas)")
                        preview_status = gr.Markdown("—")
                        preview_df = gr.Dataframe(
                            headers=None, label="Preview", interactive=False
                        )
                    with gr.Accordion("Hiperparâmetros", open=False):
                        test_size = gr.Slider(0.05, 0.5, value=0.2, step=0.05, label="test_size")
                        n_estimators = gr.Slider(50, 800, value=400, step=50, label="n_estimators")
                        max_depth = gr.Slider(0, 50, value=0, step=1, label="max_depth (0 = None)")
                    with gr.Accordion("Resultados", open=True):
                        status_md = gr.Markdown(
                            f"Modo Agents: {'✅' if AGENTS_AVAILABLE else '❌ (offline)'}"
                        )
                        metrics_json = gr.JSON(label="Métricas")
                        plot_img = gr.Image(label="Gráfico Pred vs Real", height=220)
            with gr.Row():
                run_btn = gr.Button("▶️ Rodar com as opções atuais")
                clear_btn = gr.Button("🧹 Limpar")

        with gr.TabItem("Sobre"):
            gr.Markdown(
                """
                - **Sem recursão**: 1 agente + 1 tool determinística.
                - **IDs/STORE**: arrays nunca entram no prompt nem no tracing.
                - **Imagem garantida**: sempre salva em `./artifacts/pred_vs_real.png`.
                """
            )

    # Wiring
    send_btn.click(
        fn=run_pipeline_ui,
        inputs=[user_msg, chatbot, dataset_dd, test_size, n_estimators, max_depth],
        outputs=[chatbot, metrics_json, plot_img],
        api_name="send"
    )
    run_btn.click(
        fn=run_pipeline_ui,
        inputs=[gr.Textbox(value="Execute o pipeline com as opções atuais.", visible=False),
                chatbot, dataset_dd, test_size, n_estimators, max_depth],
        outputs=[chatbot, metrics_json, plot_img],
        api_name="run"
    )
    preview_btn.click(
        fn=preview_dataset,
        inputs=[dataset_dd],
        outputs=[preview_status, preview_df],
        api_name="preview"
    )
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[chatbot, metrics_json, plot_img, status_md],
        api_name="clear"
    )

if __name__ == "__main__":
    demo.queue().launch()
