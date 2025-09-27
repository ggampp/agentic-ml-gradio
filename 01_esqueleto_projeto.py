# pip install openai-agents-python scikit-learn pandas matplotlib
# (adicione langfuse-python se quiser tracing externo)

from agents import Agent, Runner, function_tool, handoff, guardrail

# ------------ TOOLS (stubs) ----------------

@function_tool
def mcp_list_datasets() -> list[str]:
    """Lista datasets disponíveis no MCP Data Server."""
    # TODO: conectar ao MCP server e retornar nomes/ids
    return ["boston_housing", "synthetic_regression"]

@function_tool
def mcp_load_dataset(name: str) -> dict:
    """Carrega dataset do MCP (features/target)."""
    # TODO: invocar MCP e devolver {"X": [[...]], "y": [...]}
    ...

@function_tool
def py_feature_engineer(X: list[list[float]]) -> list[list[float]]:
    """Aplica transformações simples (scaling, imputations) em X."""
    ...

@function_tool
def py_train_random_forest(X: list[list[float]], y: list[float]) -> dict:
    """Treina RandomForest e retorna métricas + preds."""
    # retorne {"metrics": {"r2":..., "rmse":...}, "y_pred": [...], "y_true":[...]}
    ...

@function_tool
def py_plot_scatter(y_true: list[float], y_pred: list[float]) -> str:
    """Gera scatter plot (pred vs real) e devolve caminho da imagem."""
    # salve em /tmp ou ./artifacts e retorne o filepath
    ...

# ------------ AGENTS ----------------

supervisor = Agent(
    name="Supervisor",
    instructions="""
Você é o coordenador. Entenda o pedido do usuário, escolha o dataset,
defina o alvo e delegue para os especialistas.
""",
)

data_manager = Agent(
    name="Data Manager",
    instructions="""
Selecione e carregue um dataset adequado ao pedido.
Se necessário, aplique transformações simples.
""",
    tools=[mcp_list_datasets, mcp_load_dataset, py_feature_engineer],
)

ml_expert = Agent(
    name="ML Expert",
    instructions="""
Separe treino/teste, treine um RandomForest, reporte métricas e previsões.
""",
    tools=[py_train_random_forest],
)

visualizer = Agent(
    name="Data Visualizer",
    instructions="""
Gere gráficos (ex.: pred vs real) e devolva o caminho das imagens.
""",
    tools=[py_plot_scatter],
)

# ------------ HANDOFFS (orquestração) ------------

# Supervisor -> Data Manager
to_data = handoff(
    source=supervisor,
    target=data_manager,
    when=lambda msg: "carregar" in msg.content.lower() or True
)

# Data Manager -> ML Expert
to_ml = handoff(
    source=data_manager,
    target=ml_expert,
    when=lambda state: "X" in state.shared and "y" in state.shared
)

# ML Expert -> Visualizer
to_viz = handoff(
    source=ml_expert,
    target=visualizer,
    when=lambda state: "y_pred" in state.shared and "y_true" in state.shared
)

# Visualizer -> Supervisor (fechamento)
back_to_supervisor = handoff(
    source=visualizer,
    target=supervisor,
    when=lambda state: "plot_path" in state.shared
)

runner = Runner(
    agents=[supervisor, data_manager, ml_expert, visualizer],
    handoffs=[to_data, to_ml, to_viz, back_to_supervisor],
    # opcional: configure tracing (SDK tem "View your traces" no quickstart)
)

if __name__ == "__main__":
    print(runner.run("Quero treinar um modelo de regressão e ver o scatter plot."))
