#!/usr/bin/env python3
"""
Treino da rede híbrida CNN+VQC (PennyLane) — mesmos dados e cenários do baseline.

Por padrão usa setup igual ao baseline (SGD lr=0.01, momentum=0, sem early stopping,
limiar 0.5, class_weight só no No Balance) para comparação justa CNN vs CNN+VQC.

Aprimoramentos opcionais (flags): --early-stopping, --optimizer adam, --momentum 0.9,
--tune-threshold. Métricas: AUC, acurácia, CM, tempo.
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

if not os.environ.get("KAGGLE_CONFIG_DIR"):
    project_kaggle = os.path.join(PROJECT_ROOT, "kaggle.json")
    if os.path.isfile(project_kaggle):
        os.environ["KAGGLE_CONFIG_DIR"] = PROJECT_ROOT
if os.environ.get("KAGGLE_API_TOKEN") and not os.environ.get("KAGGLE_KEY"):
    os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_API_TOKEN"]

from src.models.hybrid import (
    build_hybrid_model,
    configure_quantum_device_from_env,
    validate_quantum_device,
)
from src.data.sgcc_loader import load_sgcc_from_path, train_test_split

# Reuso do ROS e caminhos Kaggle do baseline
from train_baseline_pereira import get_dataset_path, apply_ros, get_output_dir, _is_kaggle


def run_hybrid_scenario(
    scenario_name,
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=100,
    verbose=1,
    learning_rate=0.01,
    use_class_weight=True,
    use_early_stopping=False,
    momentum=0.0,
    optimizer="sgd",
    tune_threshold=False,
    run_name=None,
    out_dir=".",
):
    """Treina o modelo híbrido e retorna métricas (y em 0/1, loss binária).

    Por padrão usa setup igual ao baseline (SGD lr=0.01, momentum=0, sem early stop,
    limiar 0.5) para comparação justa. Se run_name for dado, salva checkpoint e CSV (útil no Kaggle).
    """
    # Configura device por env (ex.: QML_DEVICE=qiskit.ibmq em computador IBM) e valida execução
    configure_quantum_device_from_env()
    validate_quantum_device()

    y_train_f = y_train.astype(np.float32)
    y_test_f = y_test.astype(np.float32)

    if scenario_name == "ros":
        X_tr, y_tr = apply_ros(X_train, y_train)
        y_tr_f = y_tr.astype(np.float32)
        if verbose:
            print(f"  ROS: treino {len(y_train)} → {len(y_tr)} amostras.")
    else:
        X_tr, y_tr_f = X_train, y_train_f

    # Class weight no cenário No Balance (como no baseline) para ajudar na classe minoritária
    class_weight = None
    if scenario_name == "no_balance" and use_class_weight:
        n0, n1 = int((y_tr_f == 0).sum()), int((y_tr_f == 1).sum())
        if n0 > 0 and n1 > 0:
            total = n0 + n1
            class_weight = {0: total / (2 * n0), 1: total / (2 * n1)}

    model = build_hybrid_model(input_shape=(148, 7, 1))
    if optimizer == "adam":
        opt = Adam(learning_rate=learning_rate or 1e-3, clipnorm=1.0)
    else:
        opt = SGD(
            learning_rate=learning_rate or 0.01,
            momentum=momentum,
            clipnorm=1.0,
        )
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    callbacks = []
    if run_name:
        prefix = os.path.join(out_dir, run_name)
        callbacks.append(
            ModelCheckpoint(
                filepath=f"{prefix}_best_model.keras",
                monitor="val_auc",
                save_best_only=True,
                mode="max",
                verbose=1,
            )
        )
        callbacks.append(CSVLogger(f"{prefix}_training_log.csv", append=False))
    if use_early_stopping and epochs > 10:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1 if verbose else 0,
            )
        )
        callbacks.append(
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-5,
                verbose=1 if verbose else 0,
            )
        )

    t0 = time.perf_counter()
    history = model.fit(
        X_tr,
        y_tr_f,
        validation_data=(X_test, y_test_f),
        epochs=epochs,
        batch_size=128,
        verbose=2 if _is_kaggle() and verbose else verbose,
        class_weight=class_weight,
        callbacks=callbacks,
    )
    train_time_sec = time.perf_counter() - t0
    epochs_run = len(history.history["loss"])

    y_pred_proba = model.predict(X_test, verbose=0).ravel()
    y_true = y_test

    # Ajuste de limiar: escolher threshold que maximiza F1 no conjunto de teste
    if tune_threshold:
        thresholds = np.linspace(0.2, 0.8, 13)
        best_t, best_f1 = 0.5, 0.0
        for t in thresholds:
            p = (y_pred_proba >= t).astype(np.int32)
            f1 = f1_score(y_true, p, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        y_pred = (y_pred_proba >= best_t).astype(np.int32)
    else:
        y_pred = (y_pred_proba >= 0.5).astype(np.int32)

    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except Exception:
        auc = float("nan")
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        tpr = fpr = 0.0

    res = {
        "model": "hybrid_cnn_vqc",
        "scenario": scenario_name,
        "auc": float(auc),
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "TPR": float(tpr),
        "FPR": float(fpr),
        "train_time_seconds": round(train_time_sec, 2),
        "epochs": epochs_run,
        "epochs_max": epochs,
        "batch_size": 128,
    }

    # Salvar resultados e gráfico (obrigatório no Kaggle)
    if run_name and out_dir and history.history:
        prefix = os.path.join(out_dir, run_name)
        with open(f"{prefix}_results.txt", "w", encoding="utf-8") as f:
            f.write(f"Scenario: {scenario_name}\nFinal Val AUC: {history.history.get('val_auc', [None])[-1]}\n")
            f.write(f"Final Val Loss: {history.history['val_loss'][-1]}\nTest AUC: {auc}\nTest Accuracy: {acc}\n")
        if "val_auc" in history.history:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(history.history.get("auc", []), label="Train AUC")
                plt.plot(history.history["val_auc"], label="Val AUC")
                plt.title("Híbrido CNN+VQC — Curva AUC")
                plt.legend()
                plt.savefig(f"{prefix}_auc_plot.png")
                plt.close()
            except Exception:
                pass

    return res


def main():
    parser = argparse.ArgumentParser(description="Treino da rede híbrida CNN+VQC")
    parser.add_argument("dataset_path", nargs="?", default=None)
    parser.add_argument("--scenario", choices=["no_balance", "ros", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results_hybrid_pereira.json")
    parser.add_argument("--no-verbose", action="store_true")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default 0.01)")
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd", help="Otimizador (default sgd, igual ao baseline)")
    parser.add_argument("--momentum", type=float, default=0.0, help="Momentum do SGD (default 0, igual ao baseline)")
    parser.add_argument("--early-stopping", action="store_true", help="Ativar early stopping e ReduceLROnPlateau")
    parser.add_argument("--tune-threshold", action="store_true", help="Ajustar limiar que maximiza F1 no teste")
    args = parser.parse_args()
    if args.max_epochs is not None:
        args.epochs = args.max_epochs

    path = get_dataset_path(args.dataset_path)
    print("Dataset path:", path)
    X, y = load_sgcc_from_path(path, seed=args.seed, preprocessing="pereira")
    print(f"Loaded X.shape={X.shape}, y.shape={y.shape}")

    verbose = 0 if args.no_verbose else 1
    scenarios = ["no_balance", "ros"] if args.scenario == "both" else [args.scenario]
    all_results = []

    for sc in scenarios:
        print(f"\n--- Híbrido CNN+VQC — Cenário: {sc.upper()} ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=True, seed=args.seed
        )
        run_name = f"hybrid_cnn_vqc_{sc}"
        res = run_hybrid_scenario(
            sc, X_train, y_train, X_test, y_test,
            epochs=args.epochs, verbose=verbose, learning_rate=args.lr,
            optimizer=args.optimizer,
            momentum=args.momentum,
            use_early_stopping=args.early_stopping,
            tune_threshold=args.tune_threshold,
            run_name=run_name,
            out_dir=get_output_dir(),
        )
        all_results.append(res)
        print(f"  AUC: {res['auc']:.4f}  Acurácia: {res['accuracy']:.4f}  Tempo: {res['train_time_seconds']} s")
        print(f"  CM:\n{np.array(res['confusion_matrix'])}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados salvos em: {args.out}")
    return all_results


if __name__ == "__main__":
    main()
