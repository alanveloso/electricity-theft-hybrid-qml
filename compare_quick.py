#!/usr/bin/env python3
"""
Comparação rápida: Baseline CNN vs Híbrido CNN+VQC (poucas épocas).

Cenários (aplicados igual aos dois modelos):
  - no_balance: treino nos dados desbalanceados (sem oversampling).
  - ros: treino com Random Oversampling (apply_ros) nos dados de treino.

Imprime tabela com AUC, Acurácia e Tempo. Referência (Pereira & Saraiva): No Balance AUC ~0,52 Acc ~91,6% | ROS AUC ~0,67 Acc ~67,8%.
Use --qiskit-aer para rodar o VQC no simulador Qiskit Aer (estilo IBM).
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

if not os.environ.get("KAGGLE_CONFIG_DIR"):
    p = os.path.join(PROJECT_ROOT, "kaggle.json")
    if os.path.isfile(p):
        os.environ["KAGGLE_CONFIG_DIR"] = PROJECT_ROOT
if os.environ.get("KAGGLE_API_TOKEN") and not os.environ.get("KAGGLE_KEY"):
    os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_API_TOKEN"]

import numpy as np
from src.data.sgcc_loader import load_sgcc_from_path, train_test_split
from train_baseline_pereira import get_dataset_path, run_scenario as run_baseline_scenario
from train_hybrid_pereira import run_hybrid_scenario


def main():
    parser = argparse.ArgumentParser(description="Comparação rápida Baseline vs Híbrido CNN+VQC")
    parser.add_argument("--qiskit-aer", action="store_true", help="Usar simulador qiskit.aer (IBM) para o VQC")
    args = parser.parse_args()

    if args.qiskit_aer:
        try:
            from src.models.hybrid import set_quantum_device
            set_quantum_device(device="qiskit.aer")
            print("VQC usando simulador qiskit.aer (IBM)\n")
        except Exception as e:
            print("ERRO ao usar qiskit.aer:", e)
            print("Instale: pip install pennylane-qiskit\n")
            return 1

    epochs = 5   # Rápido para comparação; use 100 para resultados próximos do artigo
    seed = 42
    verbose = 0

    path = get_dataset_path(None)
    print("Carregando dados...")
    X, y = load_sgcc_from_path(path, seed=seed, preprocessing="pereira")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=True, seed=seed
    )
    print(f"N = {len(X)}, treino = {len(X_train)}, teste = {len(X_test)}\n")

    results = []

    for scenario in ["no_balance", "ros"]:
        # Baseline
        res_b, _, _ = run_baseline_scenario(
            scenario, X_train, y_train, X_test, y_test,
            epochs=epochs, verbose=verbose, learning_rate=0.01, use_class_weight=False,
        )
        results.append(("Baseline (CNN)", scenario, res_b["auc"], res_b["accuracy"], res_b["train_time_seconds"]))

        # Híbrido
        res_h = run_hybrid_scenario(
            scenario, X_train, y_train, X_test, y_test,
            epochs=epochs, verbose=verbose, learning_rate=0.01,
        )
        results.append(("Híbrido (CNN+VQC)", scenario, res_h["auc"], res_h["accuracy"], res_h["train_time_seconds"]))

    # Tabela
    print("=" * 80)
    print("COMPARAÇÃO RÁPIDA ({} épocas, 1 run, seed={})".format(epochs, seed))
    print("=" * 80)
    print(f"{'Modelo':<20} {'Cenário':<12} {'AUC':>8} {'Acurácia':>10} {'Tempo (s)':>10}")
    print("-" * 80)
    for model, scenario, auc, acc, t in results:
        print(f"{model:<20} {scenario:<12} {auc:>8.4f} {acc*100:>9.2f}% {t:>10.1f}")
    print("-" * 80)
    print("Referência (artigo, 100 épocas, 10 runs):")
    print("  No Balance: AUC 0,5162 ± 0,0045  Acurácia 91,59%")
    print("  ROS:        AUC 0,6714 ± 0,0062  Acurácia 67,78%")
    print("=" * 80)


if __name__ == "__main__":
    sys.exit(main() or 0)
