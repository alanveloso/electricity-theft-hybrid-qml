#!/usr/bin/env python3
"""
Verifica se dados e modelo estão sendo carregados/construídos corretamente.

Uso (na raiz do projeto, com venv ativo):
  python scripts/verify_model_loading.py

Ou copie as verificações para uma célula do notebook, depois de carregar X, y.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
from src.data.sgcc_loader import load_sgcc_from_path, train_test_split
from train_baseline_pereira import get_dataset_path
from src.models.cnn import build_cnn
from src.models.hybrid import build_hybrid_model, configure_quantum_device_from_env, validate_quantum_device


def verify_data(X, y):
    """Verifica se o dataset foi carregado no formato esperado."""
    print("=== 1. DADOS (X, y) ===\n")
    expected_shape = (None, 148, 7, 1)
    ok_shape = X.ndim == 4 and X.shape[1:] == (148, 7, 1)
    print(f"  X.shape = {X.shape}  (esperado: (n_amostras, 148, 7, 1))  {'OK' if ok_shape else 'ERRO'}")
    print(f"  y.shape = {y.shape}  (esperado: (n_amostras,))  {'OK' if y.shape[0] == X.shape[0] else 'ERRO'}")
    print(f"  X.dtype = {X.dtype}  (esperado: float32 ou float64)")
    print(f"  Classes: Normal (0) = {(y == 0).sum()}, Fraude (1) = {(y == 1).sum()}")
    uniq = np.unique(y)
    print(f"  y contém apenas 0 e 1: {set(uniq) <= {0, 1}}  {'OK' if set(uniq) <= {0, 1} else 'ERRO'}")
    print(f"  X sem NaN: {not np.any(np.isnan(X))}  {'OK' if not np.any(np.isnan(X)) else 'ERRO'}")
    return ok_shape


def verify_baseline_model(X_sample):
    """Verifica se a CNN baseline está sendo construída corretamente."""
    print("\n=== 2. MODELO BASELINE (CNN) ===\n")
    model = build_cnn(input_shape=(148, 7, 1))
    model.compile(
        optimizer="sgd",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    # Resumo e uma predição
    model.summary()
    out = model.predict(X_sample, verbose=0)
    print(f"\n  Entrada: shape {X_sample.shape}")
    print(f"  Saída (probabilidades): shape {out.shape}  (esperado: (batch, 2))  {'OK' if out.shape == (X_sample.shape[0], 2) else 'ERRO'}")
    print(f"  Soma das probabilidades por amostra ≈ 1: {np.allclose(out.sum(axis=1), 1.0)}")
    return out.shape == (X_sample.shape[0], 2)


def verify_hybrid_model(X_sample):
    """Verifica se o modelo híbrido CNN+VQC está sendo construído corretamente."""
    print("\n=== 3. MODELO HÍBRIDO (CNN+VQC) ===\n")
    configure_quantum_device_from_env()
    validate_quantum_device()
    model = build_hybrid_model(input_shape=(148, 7, 1))
    model.compile(
        optimizer="sgd",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    out = model.predict(X_sample, verbose=0)
    out = np.asarray(out).ravel()
    print(f"\n  Entrada: shape {X_sample.shape}")
    print(f"  Saída (probabilidade classe 1): shape {out.shape}  (esperado: (batch,))  {'OK' if out.size == X_sample.shape[0] else 'ERRO'}")
    print(f"  Valores em [0,1]: {np.all((out >= 0) & (out <= 1))}")
    return out.size == X_sample.shape[0]


def verify_loaded_checkpoint(path_keras, X_sample, is_hybrid=False):
    """Verifica se um modelo salvo (.keras) carrega e prediz corretamente."""
    import tensorflow as tf
    print(f"\n=== 4. CARREGAR CHECKPOINT: {path_keras} ===\n")
    if not os.path.isfile(path_keras):
        print(f"  Arquivo não encontrado. Pule esta verificação ou treine antes.")
        return
    try:
        model = tf.keras.models.load_model(path_keras)
    except Exception as e:
        print(f"  Erro ao carregar (ex.: camadas customizadas): {e}")
        return
    model.summary()
    out = model.predict(X_sample, verbose=0)
    if is_hybrid:
        out = np.asarray(out).ravel()
        print(f"  Saída shape: {out.shape}  (esperado: (batch,))")
    else:
        print(f"  Saída shape: {out.shape}  (esperado: (batch, 2))")
    print("  Carregamento e predição: OK")


def main():
    print("Carregando dataset...")
    path = get_dataset_path(None)
    X, y = load_sgcc_from_path(path, seed=42, preprocessing="pereira")
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=True, seed=42)

    verify_data(X, y)
    X_sample = X_test[:2]  # 2 amostras para teste rápido
    verify_baseline_model(X_sample)
    verify_hybrid_model(X_sample)

    # Opcional: verificar um checkpoint se existir
    for name in ["baseline_no_balance_run1_best_model.keras", "hybrid_no_balance_run1_best_model.keras"]:
        if "hybrid" in name:
            verify_loaded_checkpoint(name, X_sample, is_hybrid=True)
        else:
            verify_loaded_checkpoint(name, X_sample, is_hybrid=False)

    print("\n=== Verificação concluída. ===")


if __name__ == "__main__":
    main()
