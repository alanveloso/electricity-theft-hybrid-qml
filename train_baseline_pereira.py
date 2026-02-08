#!/usr/bin/env python3
"""
Baseline Single CNN — Pereira & Saraiva (2021), conforme artigo.

Referência do artigo (10 execuções, split 80/20 aleatório):
  | Cenário    | Média AUC | DP (AUC) | Melhor AUC | Média Acurácia |
  | No Balance | 0,5162    | 0,0045   | 0,5235     | 91,59%         |
  | ROS        | 0,6714    | 0,0062   | 0,6813     | 67,78%         |
  (Dataset SGCC original: 42.372 usuários; bensalem14/sgcc-dataset pode ter menos.)

Pré-processamento: interpolação linear (Eq. 1), 1035→1036 dias, reshape (148, 7, 1).
Hiperparâmetros (Seção 4.3): SGD (sem momentum citado), LR conforme ref. [17], batch 128, 100 épocas,
  cross-entropy categórica, Softmax na saída. Sem class_weight em No Balance/ROS (Weighting é cenário à parte).

Uso:
  python train_baseline_pereira.py --scenario both --n-runs 10   # replicar tabela do artigo
  python train_baseline_pereira.py --scenario ros --n-runs 1    # uma execução rápida
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

# Raiz do projeto
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

if not os.environ.get("KAGGLE_CONFIG_DIR"):
    project_kaggle = os.path.join(PROJECT_ROOT, "kaggle.json")
    if os.path.isfile(project_kaggle):
        os.environ["KAGGLE_CONFIG_DIR"] = PROJECT_ROOT
if os.environ.get("KAGGLE_API_TOKEN") and not os.environ.get("KAGGLE_KEY"):
    os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_API_TOKEN"]

from src.models.cnn import build_cnn
from src.data.sgcc_loader import load_sgcc_from_path, train_test_split


def _is_kaggle():
    """True se estiver rodando no ambiente Kaggle (leitura em /kaggle/input, escrita em ./)."""
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None or os.path.exists("/kaggle/working")


def get_output_dir():
    """Diretório para salvar modelos e resultados. No Kaggle use ./ (cwd = /kaggle/working)."""
    return "."


def get_dataset_path(args_path):
    """Caminho do dataset. No Kaggle: ../input/nome-do-dataset/ (somente leitura)."""
    if args_path and os.path.isdir(args_path):
        return args_path
    env_path = os.environ.get("SGCC_DATASET_PATH", "").strip()
    if env_path and os.path.isdir(env_path):
        return env_path
    if _is_kaggle():
        # Kaggle: dataset costuma estar em ../input/<slug>/
        kaggle_input = os.environ.get("KAGGLE_INPUT_PATH", "../input/bensalem14-sgcc-dataset")
        for candidate in [kaggle_input, "../input/bensalem14-sgcc-dataset", "../input/sgcc-dataset"]:
            if os.path.isdir(candidate):
                return os.path.abspath(candidate)
        parent = os.path.join(os.getcwd(), "..", "input")
        if os.path.isdir(parent):
            subs = [os.path.join(parent, d) for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d))]
            if subs:
                return os.path.abspath(subs[0])
    try:
        import kagglehub
        return kagglehub.dataset_download("bensalem14/sgcc-dataset")
    except Exception as e:
        print("Download falhou. Use: python train_baseline_pereira.py /caminho/para/dados")
        raise SystemExit(1) from e


def apply_ros(X_train, y_train, seed=42):
    """
    Random Oversampling: duplica aleatoriamente amostras da classe minoritária
    até igualar o tamanho da classe majoritária. Apenas nos dados de treino.
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y_train, return_counts=True)
    n_maj = counts.max()
    n_min = counts.min()
    if n_maj == n_min:
        return X_train, y_train

    # Índices por classe
    idx_by_class = [np.where(y_train == c)[0] for c in classes]
    majority_class = classes[counts.argmax()]
    minority_class = classes[counts.argmin()]
    minority_idx = np.where(y_train == minority_class)[0]
    n_extra = n_maj - n_min
    # Amostrar com reposição da classe minoritária
    extra_idx = rng.choice(minority_idx, size=n_extra, replace=True)
    new_X = np.concatenate([X_train, X_train[extra_idx]], axis=0)
    new_y = np.concatenate([y_train, y_train[extra_idx]], axis=0)
    # Embaralhar
    perm = rng.permutation(len(new_y))
    return new_X[perm], new_y[perm]


def run_scenario(
    scenario_name,
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=100,
    verbose=1,
    learning_rate=0.01,
    use_class_weight=True,
    run_name=None,
    out_dir=".",
):
    """Treina a Single CNN e retorna métricas e modelo. Se run_name for dado, salva checkpoint e CSV do treino (útil no Kaggle)."""
    y_train_cat = to_categorical(y_train, num_classes=2)
    y_test_cat = to_categorical(y_test, num_classes=2)

    if scenario_name == "ros":
        X_tr, y_tr = apply_ros(X_train, y_train)
        y_tr_cat = to_categorical(y_tr, num_classes=2)
        if verbose:
            print(f"  ROS: treino {len(y_train)} → {len(y_tr)} amostras (balanceado).")
    else:
        X_tr, y_tr_cat = X_train, y_train_cat

    model = build_cnn(input_shape=(148, 7, 1))
    # SGD sem momentum (artigo não cita); LR padrão 0.01 (ref. [17] não especifica valor)
    model.compile(
        optimizer=SGD(learning_rate=learning_rate, momentum=0.0),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    # Artigo não usa class_weight em No Balance nem ROS (apenas no cenário "Weighting")
    class_weight = None
    if scenario_name == "no_balance" and use_class_weight:
        n0, n1 = (y_train == 0).sum(), (y_train == 1).sum()
        total = n0 + n1
        class_weight = {0: total / (2 * n0), 1: total / (2 * n1)}

    callbacks_list = []
    if run_name:
        prefix = os.path.join(out_dir, run_name)
        callbacks_list.append(
            ModelCheckpoint(
                filepath=f"{prefix}_best_model.keras",
                monitor="val_auc",
                save_best_only=True,
                mode="max",
                verbose=1,
            )
        )
        callbacks_list.append(CSVLogger(f"{prefix}_training_log.csv", append=False))

    t0 = time.perf_counter()
    history = model.fit(
        X_tr,
        y_tr_cat,
        validation_data=(X_test, y_test_cat),
        epochs=epochs,
        batch_size=128,
        verbose=2 if _is_kaggle() and verbose else verbose,
        class_weight=class_weight,
        callbacks=callbacks_list,
    )
    train_time_sec = time.perf_counter() - t0

    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = y_test

    try:
        auc = roc_auc_score(y_true, y_pred_proba[:, 1])
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

    results = {
        "scenario": scenario_name,
        "auc": float(auc),
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "TPR": float(tpr),
        "FPR": float(fpr),
        "train_time_seconds": round(train_time_sec, 2),
        "epochs": epochs,
        "batch_size": 128,
    }

    # Salvar resultados e gráfico (obrigatório no Kaggle para não perder nada)
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
                plt.title("CNN Pereira & Saraiva — Curva AUC")
                plt.legend()
                plt.savefig(f"{prefix}_auc_plot.png")
                plt.close()
            except Exception:
                pass

    return results, model, history


def main():
    parser = argparse.ArgumentParser(
        description="Baseline Single CNN (Pereira & Saraiva 2021) — Cenários A (No Balance) e B (ROS)"
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default=None,
        help="Pasta do dataset SGCC (opcional; senão usa env ou Kaggle)",
    )
    parser.add_argument(
        "--scenario",
        choices=["no_balance", "ros", "both"],
        default="both",
        help="Cenário: no_balance (A), ros (B), ou both (A e B)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Épocas (default: 100)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        metavar="N",
        help="Alias para --epochs (teste rápido, ex: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para split e ROS",
    )
    parser.add_argument(
        "--out",
        default="results_baseline_pereira.json",
        help="Arquivo JSON para salvar métricas",
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Treino em modo quiet (verbose=0)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="F",
        help="Learning rate do SGD (default: 0.01). Teste 0.02 ou 0.05 se AUC ficar em 0.5.",
    )
    parser.add_argument(
        "--class-weight",
        action="store_true",
        help="Usar class_weight no No Balance (cenário extra; artigo não usa em No Balance/ROS)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        metavar="N",
        help="Número de execuções com diferentes splits (default: 1). Use 10 para replicar tabela do artigo.",
    )
    args = parser.parse_args()
    if args.max_epochs is not None:
        args.epochs = args.max_epochs

    path = get_dataset_path(args.dataset_path)
    print("Dataset path:", path)
    print("Pré-processamento: interpolação linear (Eq. 1) + 1036 dias + reshape (148, 7, 1)")
    X, y = load_sgcc_from_path(path, seed=args.seed, preprocessing="pereira")
    print(f"Loaded X.shape={X.shape}, y.shape={y.shape}, classes={np.unique(y)}")
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    print(f"  Normal: {n0}, Fraude: {n1} ({100 * n1 / len(y):.1f}%)")
    print(f"  (Artigo: 42.372 usuários, 38.757 normais, 3.615 anômalos)")

    verbose = 0 if args.no_verbose else 1
    use_class_weight = args.class_weight
    scenarios = []
    if args.scenario in ("no_balance", "both"):
        scenarios.append("no_balance")
    if args.scenario in ("ros", "both"):
        scenarios.append("ros")

    all_results = []
    for sc in scenarios:
        aucs, accs, run_details = [], [], []
        for run in range(args.n_runs):
            run_seed = args.seed + run
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=True, seed=run_seed
            )
            if args.n_runs > 1 and verbose:
                print(f"\n--- Cenário: {sc.upper()} — Run {run + 1}/{args.n_runs} (seed={run_seed}) ---")
            run_name = f"cnn_pereira_{sc}" if args.n_runs == 1 else f"cnn_pereira_{sc}_run{run+1}"
            res, _, _ = run_scenario(
                sc,
                X_train,
                y_train,
                X_test,
                y_test,
                epochs=args.epochs,
                verbose=verbose if (args.n_runs == 1 or run == 0) else 0,
                learning_rate=args.lr,
                use_class_weight=use_class_weight,
                run_name=run_name,
                out_dir=get_output_dir(),
            )
            aucs.append(res["auc"])
            accs.append(res["accuracy"])
            run_details.append(
                {
                    "run": run + 1,
                    "seed": run_seed,
                    "auc": res["auc"],
                    "accuracy": res["accuracy"],
                    "confusion_matrix": res["confusion_matrix"],
                    "train_time_seconds": res["train_time_seconds"],
                }
            )

        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
        best_auc = float(np.max(aucs))
        mean_acc = float(np.mean(accs)) * 100
        summary = {
            "scenario": sc,
            "n_runs": args.n_runs,
            "mean_auc": round(mean_auc, 4),
            "std_auc": round(std_auc, 4),
            "best_auc": round(best_auc, 4),
            "mean_accuracy_pct": round(mean_acc, 2),
            "runs": run_details,
        }
        all_results.append(summary)

        if args.n_runs == 1:
            print(f"\n--- Cenário: {sc.upper()} ---")
            print(f"  AUC:        {res['auc']:.4f}")
            print(f"  Acurácia:   {res['accuracy']:.4f}")
            print(f"  Matriz de confusão:\n{np.array(res['confusion_matrix'])}")
            print(f"  TPR: {res['TPR']:.4f}, FPR: {res['FPR']:.4f}")
            print(f"  Tempo de treino: {res['train_time_seconds']} s")
        else:
            print(f"\n--- Cenário: {sc.upper()} ({args.n_runs} execuções) ---")
            print(f"  Média AUC:      {mean_auc:.4f}  (DP: {std_auc:.4f})")
            print(f"  Melhor AUC:     {best_auc:.4f}")
            print(f"  Média Acurácia: {mean_acc:.2f}%")
            print(f"  (Artigo — No Balance: AUC 0,5162 ± 0,0045, Acc 91,59% | ROS: AUC 0,6714 ± 0,0062, Acc 67,78%)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nMétricas salvas em: {args.out}")

    return all_results


if __name__ == "__main__":
    main()
