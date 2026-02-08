# Detecção de Furto de Energia com Redes Neurais Híbridas (CNN-VQC)

Este repositório contém a implementação oficial da pesquisa comparativa entre Redes Neurais Convolucionais Clássicas e uma abordagem Híbrida Quântica para detecção de anomalias em dados desbalanceados de consumo de energia (Dataset SGCC).

## 🎯 Objetivo
Avaliar se a utilização de **Quantum Machine Learning (VQC)** como classificador final permite manter alta performance (AUC) reduzindo a necessidade de técnicas agressivas de balanceamento de dados (como ROS/SMOTE), economizando memória computacional.

## 🏗️ Arquitetura
O projeto compara dois cenários de balanceamento, **em ambos os modelos** (baseline e híbrido):

| Estratégia    | Baseline (CNN)     | Híbrido (CNN+VQC)   |
|---------------|--------------------|---------------------|
| **No Balance**| Treino nos dados desbalanceados (sem oversampling). | Idem: treino nos dados desbalanceados. |
| **ROS**       | Treino com Random Oversampling (duplica amostras da classe minoritária até equilibrar). | Idem: mesmo `apply_ros` nos dados de treino. |

Ou seja: em cada estratégia, baseline e híbrido usam exatamente o mesmo tratamento dos dados; a única diferença é o modelo (CNN só vs CNN+VQC).

O VQC usa **4 qubits** por padrão (referência do artigo: No Balance AUC ~0,52, ROS AUC ~0,67). Esse valor é um bom equilíbrio: simulação rápida e compatível com hardware real. Escalar para mais qubits (ex.: 6–8) faz sentido só depois de ter resultados estáveis (híbrido próximo ou melhor que o baseline); a simulação fica bem mais lenta e o treino muito mais pesado.

**Quem é quem:** O **artigo** (Pereira & Saraiva) reporta resultados da **CNN clássica sozinha** (baseline). Neste repositório comparamos essa baseline com o **híbrido QML** (CNN + VQC). Ou seja: "resultado do artigo" = baseline (sem quântico); "QML" = nossa proposta (com circuito quântico). Na corrida rápida abaixo, o baseline do artigo ficou melhor que o QML; o objetivo do projeto é ver se o QML consegue **empatar ou ficar próximo** do baseline (não obrigatoriamente superar).

**Comparação justa com a CNN:** o híbrido está em **modo padrão**: mesma receita do baseline (SGD lr=0.01, momentum=0, sem early stopping, limiar 0.5, class_weight só no No Balance). Modelo: CNN até Dense(128) → Dense(4) → VQC → Dense(1), sem dropout/L2/gargalo extra; init do VQC em [-0.2π, 0.2π] para estabilidade. Assim a diferença de resultado reflete a arquitetura (VQC), não truques de treino.

**Aprimoramentos opcionais** (para ablation): `train_hybrid_pereira.py --early-stopping --optimizer adam --momentum 0.9 --tune-threshold`. Use o notebook ou `--scenario both` para 100 épocas.

## 🚀 Como fazer funcionar

### 1. Ambiente
```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dataset SGCC — duas opções

**Opção A: Download automático (recomendado)**  
Você não precisa baixar nada à mão; o script baixa sozinho se o Kaggle estiver configurado:

1. Crie uma conta em [Kaggle](https://www.kaggle.com) e aceite as regras do dataset [SGCC Dataset](https://www.kaggle.com/datasets/bensalem14/sgcc-dataset).
2. Autenticação (uma das opções):
   - **Arquivo:** em [Kaggle → Account → API](https://www.kaggle.com/settings), crie um token e coloque `kaggle.json` em `~/.kaggle/kaggle.json`.
   - **Variáveis de ambiente:** defina `KAGGLE_USERNAME` (seu usuário) e `KAGGLE_KEY` ou `KAGGLE_API_TOKEN` (a chave do token). O script aceita `KAGGLE_API_TOKEN` e repassa como `KAGGLE_KEY` automaticamente.
3. Rode o treino (na primeira vez o dataset será baixado automaticamente):
   ```bash
   export KAGGLE_USERNAME=seu_usuario
   export KAGGLE_API_TOKEN=sua_chave   # ou KAGGLE_KEY
   python train_sgcc_cnn.py
   ```

**Opção B: Download manual**  
Se preferir baixar pelo site:

1. Acesse [SGCC Dataset](https://www.kaggle.com/datasets/bensalem14/sgcc-dataset) e clique em **Download**.
2. Descompacte o ZIP numa pasta (ex.: `data/sgcc-dataset`).
3. Rode o treino apontando para essa pasta:
   ```bash
   python train_sgcc_cnn.py data/sgcc-dataset
   ```
   Ou use a variável de ambiente:
   ```bash
   SGCC_DATASET_PATH=data/sgcc-dataset python train_sgcc_cnn.py
   ```

### 3. Treino

**Baseline Single CNN (Pereira & Saraiva 2021)** — reprodução exata do artigo para comparação:
- Pré-processamento: interpolação linear (Eq. 1), 1035→1036 dias, reshape (148, 7, 1).
- Hiperparâmetros: SGD, batch 128, 100 épocas.
- Cenários: **A (No Balance)** ou **B (ROS)**. Métricas: AUC, acurácia, matriz de confusão, tempo.
  ```bash
  python train_baseline_pereira.py --scenario no_balance   # Cenário A
  python train_baseline_pereira.py --scenario ros         # Cenário B
  python train_baseline_pereira.py --scenario both         # A e B (salva em results_baseline_pereira.json)
  python train_baseline_pereira.py --scenario both --max-epochs 3  # Teste rápido
  ```

**Script legado (Adam, EarlyStopping):** `python train_sgcc_cnn.py` — modelo em `checkpoints/best_cnn_sgcc.keras`.

### Rodando no Kaggle
- **Entrada:** o dataset deve estar em `../input/nome-do-dataset/`. O script detecta o ambiente Kaggle e tenta `../input/bensalem14-sgcc-dataset` ou a primeira pasta em `../input/`; você pode definir `KAGGLE_INPUT_PATH` se o nome for outro.
- **Saída:** modelos, logs e gráficos são salvos em `./` (diretório de trabalho = `/kaggle/working`). Em cada execução são gerados: `{run_name}_best_model.keras`, `{run_name}_training_log.csv`, `{run_name}_results.txt` e `{run_name}_auc_plot.png`.
- **Callbacks:** `ModelCheckpoint` (melhor modelo por `val_auc`) e `CSVLogger` (histórico por época) são adicionados automaticamente quando há `run_name` (o script de treino já passa isso). Assim, se o job travar no meio, você mantém o melhor modelo e o log até a última época.

### Simulador e hardware IBM (opcional)
O circuito VQC roda por padrão no simulador PennyLane (`default.qubit`). Você pode usar:

- **Simulador local no estilo IBM (Qiskit Aer)** — não precisa de token:
  ```python
  from src.models.hybrid import set_quantum_device
  set_quantum_device(device="qiskit.aer")  # pip install pennylane-qiskit
  ```
  Os simuladores **na nuvem** da IBM (ex. `ibmq_qasm_simulator`) foram desativados em 2024; o `qiskit.aer` roda na sua máquina e usa a mesma stack Qiskit/IBM. Para testar localmente: `python test_qiskit_aer.py` (requer `pip install pennylane-qiskit`).

- **Hardware real da IBM** — precisa de conta e token em [IBM Quantum](https://quantum.ibm.com):
  ```python
  set_quantum_device(device="qiskit.ibmq", backend="ibm_brisbane", ibmqx_token="SEU_TOKEN")
  # ou export IBMQX_TOKEN=... e omitir ibmqx_token
  ```
  Instale: `pip install pennylane-qiskit qiskit-ibm-runtime`. Os backends disponíveis aparecem no painel da IBM; o treino pode ter fila e limites de uso.

**Execução em computador IBM (ambiente IBM):** o código pode ser configurado só por variáveis de ambiente; antes do treino híbrido o device é validado. Defina no ambiente onde for rodar:
- `QML_DEVICE=qiskit.ibmq` (ou `qiskit.aer` para simulador local)
- `QML_IBMQ_BACKEND=ibm_brisbane` (ou o backend desejado)
- `IBMQX_TOKEN=seu_token`
Assim que o treino híbrido começar, `configure_quantum_device_from_env()` e `validate_quantum_device()` garantem que o device está pronto; se algo falhar, o erro aparece antes do treino longo.

### Teste completo no Google Colab (GPU)
Para rodar o **teste completo** (3 runs baseline, 3 runs híbrido, 100 épocas) no Colab com GPU:

1. Abra o notebook **[notebooks/colab_full_test.ipynb](notebooks/colab_full_test.ipynb)** no Google Colab (upload ou abrir via GitHub).
2. Ative **Runtime → Change runtime type → GPU**.
3. Configure as credenciais do Kaggle (Secrets do Colab com `KAGGLE_USERNAME` e `KAGGLE_KEY`, ou upload do `kaggle.json`).
4. Execute as células em ordem. O notebook clona o repositório, instala dependências, baixa o SGCC e roda baseline e híbrido, gerando uma tabela de comparação e um CSV.

### Verificar só a arquitetura da CNN (sem dados)
```bash
python -c "from src.models.cnn import build_paper_cnn; m = build_paper_cnn(); m.summary()"
```

---

1. Clone o repositório:
   ```bash
   git clone [https://github.com/alanveloso/electricity-theft-hybrid-qml.git](https://github.com/alanveloso/electricity-theft-hybrid-qml.git)