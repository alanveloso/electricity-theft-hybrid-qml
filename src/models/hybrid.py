"""Rede híbrida CNN + VQC (Variational Quantum Circuit) para detecção de furto de energia.

Substitui a camada final Dense(2, softmax) da CNN por um circuito quântico (Angle Embedding
+ Strongly Entangling Layers) rodando no simulador PennyLane. Saída binária via Dense(1, sigmoid).

Uso com simulador ou hardware IBM (opcional): instale `pennylane-qiskit`. Antes de construir o modelo:
  - Simulador local estilo IBM (Qiskit Aer, sem token): set_quantum_device(device="qiskit.aer")
  - Hardware real IBM (precisa token): set_quantum_device(device="qiskit.ibmq", backend="ibm_brisbane", ibmqx_token="...")
Nota: os simuladores na nuvem da IBM (ex. ibmq_qasm_simulator) foram desativados em 2024; use qiskit.aer para simulação local.
"""
import os
import numpy as np
import pennylane as qml
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Layer

# --- CONFIGURAÇÃO DO CIRCUITO QUÂNTICO ---
# 4 qubits: bom equilíbrio entre capacidade e custo (simulação rápida, cabe em hardware real).
# Escalar (6–8 qubits) só se: (1) híbrido já empata/supera baseline em AUC e (2) quiser testar se mais qubits melhoram; simulação fica bem mais lenta.
n_qubits = 4   # Gargalo: saída da CNN é reduzida para n_qubits (Dense(n_qubits) antes do VQC)
n_layers = 3   # Profundidade do ansatz (ajustável)

# Device e QNode configuráveis (simulador por padrão; pode trocar para IBM via set_quantum_device)
_device = None
_qnode = None


def _circuit(inputs, weights):
    """Circuito: AngleEmbedding + StronglyEntanglingLayers + medição PauliZ(0). (Múltiplas medições conflitam com TF em map_fn.)"""
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))


def _get_device():
    """Retorna o device atual (cria default.qubit se ainda não foi definido)."""
    global _device
    if _device is None:
        _device = qml.device("default.qubit", wires=n_qubits)
    return _device


def _get_qnode():
    """Retorna o QNode atual (cria com device atual se necessário)."""
    global _qnode
    if _qnode is None:
        dev = _get_device()
        _qnode = qml.qnode(dev, interface="tf")(_circuit)
    return _qnode


def set_quantum_device(
    device="default.qubit",
    backend=None,
    ibmqx_token=None,
    *,
    use_ibm=None,
):
    """Escolhe onde rodar o circuito: simulador padrão, simulador Qiskit Aer (IBM) ou hardware IBM.

    Parâmetros
    ----------
    device : str
        - "default.qubit" (padrão): simulador PennyLane em CPU.
        - "qiskit.aer": simulador local Qiskit Aer (stack IBM, não precisa de token).
        - "qiskit.ibmq": hardware real da IBM (requer token e backend).
    backend : str, opcional
        Só para device="qiskit.ibmq": nome do backend (ex. "ibm_brisbane").
        Os simuladores na nuvem da IBM foram desativados em 2024; use "qiskit.aer" para simulação.
    ibmqx_token : str, opcional
        Token da IBM (https://quantum.ibm.com). Pode usar variável de ambiente IBMQX_TOKEN.
    use_ibm : bool, opcional
        Obsoleto. Use device="qiskit.ibmq" ou device="qiskit.aer". Se True, equivale a device="qiskit.ibmq".
    """
    global _device, _qnode
    _qnode = None
    if use_ibm is not None:
        device = "qiskit.ibmq" if use_ibm else "default.qubit"
        if use_ibm and backend is None:
            backend = "ibmq_qasm_simulator"  # legado; hardware real usa outro nome
    if device == "default.qubit":
        _device = qml.device("default.qubit", wires=n_qubits)
        return
    if device == "qiskit.aer":
        try:
            _device = qml.device("qiskit.aer", wires=n_qubits)
        except Exception as e:
            raise RuntimeError(
                "Falha ao usar qiskit.aer. Instale: pip install pennylane-qiskit"
            ) from e
        return
    if device == "qiskit.ibmq":
        try:
            token = ibmqx_token or os.environ.get("IBMQX_TOKEN")
            kwargs = {"wires": n_qubits}
            if backend:
                kwargs["backend"] = backend
            if token:
                kwargs["ibmqx_token"] = token
            _device = qml.device("qiskit.ibmq", **kwargs)
        except Exception as e:
            raise RuntimeError(
                "Falha ao usar device IBM. Instale: pip install pennylane-qiskit qiskit-ibm-runtime. "
                "Token em https://quantum.ibm.com. Para simulador local estilo IBM use device='qiskit.aer'."
            ) from e
        return
    raise ValueError(
        "device deve ser 'default.qubit', 'qiskit.aer' ou 'qiskit.ibmq'"
    )


def configure_quantum_device_from_env():
    """Configura o device a partir de variáveis de ambiente (para execução em ambiente IBM).

    Variáveis lidas:
    - QML_DEVICE: "default.qubit" | "qiskit.aer" | "qiskit.ibmq"
    - QML_IBMQ_BACKEND: nome do backend (ex. "ibm_brisbane"), só para device=qiskit.ibmq
    - IBMQX_TOKEN: token IBM Quantum (ou já configurado no ambiente)

    Se QML_DEVICE não estiver definida, não altera o device atual.
    """
    device = os.environ.get("QML_DEVICE", "").strip().lower()
    if not device:
        return
    backend = os.environ.get("QML_IBMQ_BACKEND", "").strip() or None
    token = os.environ.get("IBMQX_TOKEN", "").strip() or None
    set_quantum_device(device=device, backend=backend, ibmqx_token=token)


def validate_quantum_device(*, run_dry_circuit=False):
    """Valida que o device está pronto para executar (ex.: antes do treino em computador IBM).

    Garante que o device foi criado e, se for qiskit.ibmq, que há token/backend.
    Opcionalmente executa um circuito mínimo (run_dry_circuit=True) para checar conectividade.

    Raises
    ------
    RuntimeError
        Se o device não estiver configurado corretamente ou não for possível executar.
    """
    global _device, _qnode
    dev = _get_device()
    if run_dry_circuit:
        try:
            qn = _get_qnode()
            x = np.zeros(n_qubits, dtype=np.float32)
            w = np.zeros((n_layers, n_qubits, 3), dtype=np.float32)
            _ = qn(x, w)
        except Exception as e:
            raise RuntimeError(f"Validação do device falhou (circuito de teste): {e}") from e
    return True


def qnode(inputs, weights):
    """Wrapper que delega para o QNode atual (simulador ou IBM)."""
    return _get_qnode()(inputs, weights)


class QuantumLayer(Layer):
    """Camada Keras que envolve o QNode (PennyLane não exporta KerasLayer em versões recentes)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # Inicialização em escala pequena (perto da identidade) para reduzir barren plateaus
        self.weights_var = self.add_weight(
            name="weights",
            shape=(n_layers, n_qubits, 3),
            initializer=tf.keras.initializers.RandomUniform(-0.2 * np.pi, 0.2 * np.pi),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, 4). QNode retorna um escalar (expval PauliZ(0)).
        x = tf.cast(inputs, tf.float32)
        w = tf.cast(self.weights_var, tf.float32)
        batch_size = tf.shape(x)[0]
        out = tf.map_fn(
            lambda i: tf.cast(tf.reshape(qnode(x[i], w), ()), tf.float32),
            tf.range(batch_size),
            fn_output_signature=tf.float32,
        )
        return tf.reshape(out, (-1, 1))


def get_quantum_layer():
    """Retorna a camada quântica compatível com Keras. Saída: (batch, 1), valores em [-1, 1]."""
    return QuantumLayer(name="vqc")


def build_hybrid_model(input_shape=(148, 7, 1)):
    """
    Rede híbrida: mesma CNN do baseline (Pereira & Saraiva) até Dense(128);
    depois Dense(4, tanh) → circuito quântico → Dense(1, sigmoid) para classificação binária.
    """
    model = Sequential(name="hybrid_cnn_vqc")

    # Blocos Conv + Pool (idênticos ao baseline)
    model.add(
        Conv2D(
            16,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            input_shape=input_shape,
            name="conv2d_16",
        )
    )
    model.add(MaxPooling2D(pool_size=(4, 1), padding="same", name="maxpool_1"))
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same", name="conv2d_32"))
    model.add(MaxPooling2D(pool_size=(4, 1), padding="same", name="maxpool_2"))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2d_64"))
    model.add(MaxPooling2D(pool_size=(3, 2), padding="same", name="maxpool_3"))

    model.add(Flatten(name="flatten"))
    model.add(Dense(128, activation="relu", name="dense_128"))
    # Gargalo: 4 features para os 4 qubits (igual ao baseline até aqui, depois VQC)
    model.add(Dense(n_qubits, activation="tanh", name="dense_4"))

    # Camada quântica variacional (VQC)
    model.add(get_quantum_layer())

    # Saída binária: probabilidade de fraude
    model.add(Dense(1, activation="sigmoid", name="output"))

    return model
