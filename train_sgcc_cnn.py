#!/usr/bin/env python3
"""
Treina a CNN do paper no dataset SGCC (Kaggle).

Uso:
  python train_sgcc_cnn.py                    # baixa via Kaggle API (precisa de credenciais)
  python train_sgcc_cnn.py --max-epochs 3    # teste rápido (3 épocas)
  python train_sgcc_cnn.py /caminho/para/dados # usa pasta local (ex.: após download manual)
  SGCC_DATASET_PATH=/caminho python train_sgcc_cnn.py  # idem, via variável de ambiente
"""
import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Raiz do projeto
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Kaggle: usar kaggle.json da pasta do projeto se existir (senão usa ~/.kaggle/)
if not os.environ.get("KAGGLE_CONFIG_DIR"):
    project_kaggle = os.path.join(PROJECT_ROOT, "kaggle.json")
    if os.path.isfile(project_kaggle):
        os.environ["KAGGLE_CONFIG_DIR"] = PROJECT_ROOT

# Se você usa export KAGGLE_API_TOKEN=..., o Kaggle API espera KAGGLE_KEY
if os.environ.get("KAGGLE_API_TOKEN") and not os.environ.get("KAGGLE_KEY"):
    os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_API_TOKEN"]

from src.models.cnn import build_paper_cnn
from src.data.sgcc_loader import load_sgcc_from_path, train_test_split


def get_dataset_path(args_path):
    """Retorna o path dos dados: argumento > env SGCC_DATASET_PATH > download via kagglehub."""
    if args_path:
        if not os.path.isdir(args_path):
            print(f"Erro: pasta não encontrada: {args_path}")
            sys.exit(1)
        return args_path
    env_path = os.environ.get("SGCC_DATASET_PATH", "").strip()
    if env_path and os.path.isdir(env_path):
        return env_path
    # Download automático via Kaggle
    try:
        import kagglehub
        return kagglehub.dataset_download("bensalem14/sgcc-dataset")
    except Exception as e:
        print("Download automático falhou. Verifique:")
        print("  - KAGGLE_USERNAME e KAGGLE_KEY (ou KAGGLE_API_TOKEN) ou ~/.kaggle/kaggle.json")
        print("  - Regras do dataset aceitas em https://www.kaggle.com/datasets/bensalem14/sgcc-dataset")
        print("Alternativa: baixe o dataset manualmente e rode: python train_sgcc_cnn.py /caminho/para/pasta")
        raise SystemExit(1) from e


def main():
    parser = argparse.ArgumentParser(description="Treina CNN no dataset SGCC")
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default=None,
        help="Pasta com os CSVs do SGCC (opcional; senão usa SGCC_DATASET_PATH ou baixa via Kaggle)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        metavar="N",
        help="Número máximo de épocas (default: 100). Use 2–3 para teste rápido.",
    )
    args = parser.parse_args()

    path = get_dataset_path(args.dataset_path)
    print("Dataset path:", path)

    print("Loading and reshaping data to (148, 7, 1)...")
    X, y = load_sgcc_from_path(path)
    print(f"Loaded X.shape={X.shape}, y.shape={y.shape}, classes={np.unique(y)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
    y_train_cat = to_categorical(y_train, num_classes=2)
    y_test_cat = to_categorical(y_test, num_classes=2)

    model = build_paper_cnn(input_shape=(148, 7, 1))
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    os.makedirs("checkpoints", exist_ok=True)
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            "checkpoints/best_cnn_sgcc.keras",
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    print(f"Training (max_epochs={args.max_epochs})...")
    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=args.max_epochs,
        batch_size=32,
        callbacks=callbacks,
    )

    loss, acc = model.evaluate(X_test, y_test_cat)
    print(f"Test accuracy: {acc:.4f}, Test loss: {loss:.4f}")
    return model, history


if __name__ == "__main__":
    main()
