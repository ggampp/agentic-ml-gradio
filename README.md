

# Agentic ML Pipeline

> **Aprendizado de Máquina com Agentes + Interface Gradio (IDs, sem recursão)**

Este projeto demonstra como orquestrar um *pipeline* de Machine Learning usando **agentes inteligentes** (OpenAI Agents SDK) e também uma execução **offline 100% determinística**.
Ele inclui:

* **Runner offline** com múltiplos papéis (Supervisor, Data Manager, ML Expert, Data Visualizer)
* **Runner com Agents SDK** usando **apenas uma tool determinística** (`pipeline_run`)
* **Interface Gradio** rica, em formato de **chatbot interativo**, trafegando **apenas IDs** para evitar estouro de contexto.

---

## 📂 Estrutura do Projeto

| Arquivo                      | Descrição                                                                       |
| ---------------------------- | ------------------------------------------------------------------------------- |
| **01_esqueleto_projeto.py**  | Esqueleto inicial do projeto com *stubs* de agents e tools.                     |
| **02_agentic_ml_runner.py**  | Runner offline 100% executável, sem dependência de LLM.                         |
| **03_runner_agents_sdk.py**  | Runner usando **OpenAI Agents SDK** com uma única tool `pipeline_run`.          |
| **04_app_gradio_agentic.py** | Interface **Gradio** em modo chatbot, trafegando apenas **IDs** e sem recursão. |
| **requirements.txt**         | Lista mínima de dependências.                                                   |

---

## 🔑 Principais Recursos

* **Datasets integrados**:

  * `diabetes` (scikit-learn)
  * `california_housing` (scikit-learn)
  * `synthetic_regression` (gerado com `make_regression`)
* **Pipeline de ML**:

  1. Carregamento do dataset
  2. *Feature Engineering* (imputação + `StandardScaler`)
  3. Treino de `RandomForestRegressor`
  4. Avaliação (R², RMSE) + geração de *scatter plot*
* **Armazenamento por IDs**: evita tráfego de arrays gigantes no contexto do LLM.
* **Execução determinística**: a imagem e métricas são sempre geradas no diretório `./artifacts`.
* **Interface visual**: chatbot, preview de dados, sliders de hiperparâmetros, exibição de métricas e imagem.

---

## ⚙️ Requisitos

Arquivo `requirements.txt`:

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
openai-agents>=0.1.0   # apenas para os runners com Agents/Gradio
```



Instale com:

```bash
pip install -r requirements.txt
```

---

## 🚀 Modos de Execução

### 1. **Runner Offline** (sem chave de API)

Executa todo o pipeline localmente, ideal para testes rápidos.

```bash
python 02_agentic_ml_runner.py --task "Treinar um modelo de regressão" --dataset diabetes
```

Saída:

* Métricas (R² e RMSE) no terminal
* Gráfico `pred_vs_real.png` em `./artifacts`

---

### 2. **Runner com OpenAI Agents SDK**

Requer variável `OPENAI_API_KEY` configurada.

```bash
export OPENAI_API_KEY="sua_chave_aqui"  # Windows: set OPENAI_API_KEY=...
python 03_runner_agents_sdk.py --dataset california_housing --n_estimators 300
```

O agente chama **apenas uma tool** (`pipeline_run`) e retorna um JSON final com métricas e caminho do gráfico.

---

### 3. **Interface Gradio (Chatbot)**

Interface web interativa, suportando:

* Seleção de dataset
* Sliders para hiperparâmetros
* Preview de 25 linhas do dataset
* Geração de métricas e gráfico

```bash
export OPENAI_API_KEY="sua_chave_aqui"  # opcional
python 04_app_gradio_agentic.py
```

Acesse o endereço exibido (ex.: `http://127.0.0.1:7860`) .

> ⚠️ Se não houver chave de API, o app entra em **modo offline** automaticamente.

---

## 🖼️ Resultados

* **Métricas**: R² e RMSE calculados no *hold-out set*.
* **Gráfico**: `./artifacts/pred_vs_real.png` mostrando valores previstos vs. reais.

---

## 💡 Extensões Sugeridas

* Adicionar novos datasets (ex.: Ames Housing, Bike Sharing) seguindo o padrão de *loaders* por IDs.
* Conectar a um MCP Data Server real.
* Integrar outros algoritmos (XGBoost, LightGBM) ou tarefas (classificação).

---

## 📜 Licença

Este projeto é disponibilizado sob a licença MIT.
Use livremente para estudos, demonstrações ou como base para projetos próprios.

---

Pronto! Com este README, qualquer pessoa consegue:

1. Instalar as dependências,
2. Rodar o pipeline offline,
3. Usar a versão com Agents SDK,
4. Experimentar a interface Gradio em modo chatbot.
