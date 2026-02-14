"""Carrega e prepara o dataset SGCC para o formato (148, 7, 1) esperado pela CNN.

Pré-processamento Pereira & Saraiva (2021):
- Valores ausentes: interpolação linear (Eq. 1 do artigo).
- Dados originais 1035 dias → adicionar 1 dia (cópia do último) → 1036 dias.
- Reshape: vetor 1D → matriz (148, 7) = 148 semanas × 7 dias → (148, 7, 1).
"""
import os
import numpy as np
import pandas as pd


# Formato esperado pela CNN: (amostras, 148, 7, 1)
TIME_STEPS = 148   # 148 semanas
FEATURES = 7       # 7 dias por semana
NEEDED_DAYS = TIME_STEPS * FEATURES  # 1036 dias
ORIGINAL_DAYS = 1035  # SGCC original; adicionamos 1 para fechar 1036


def _find_csv_and_label_column(path):
    """Encontra CSVs no path e tenta identificar coluna de label e de consumo."""
    csvs = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith('.csv'):
                csvs.append(os.path.join(root, f))
    if not csvs:
        return None, None, None

    # Preferir arquivo que pareça principal (train ou maior)
    def prefer(f):
        name = os.path.basename(f).lower()
        if 'train' in name:
            return 0
        if 'test' in name:
            return 1
        return 2
    csvs.sort(key=prefer)
    csv_path = csvs[0]

    df = pd.read_csv(csv_path, nrows=5)
    cols = list(df.columns)

    # Label: colunas comuns em datasets de theft
    label_candidates = [c for c in cols if any(x in c.lower() for x in ('label', 'target', 'class', 'flag', 'fraud', 'theft'))]
    label_col = label_candidates[0] if label_candidates else None

    # Colunas numéricas de consumo (D1, D2, ... ou day_, consumption, etc.)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col and label_col in numeric:
        numeric.remove(label_col)
    # Remover IDs
    id_like = [c for c in numeric if any(x in c.lower() for x in ('id', 'index', 'num'))]
    for c in id_like:
        if c in numeric:
            numeric.remove(c)
    # Ordenar para ter ordem temporal (D1, D2, ... ou similar)
    day_cols = [c for c in numeric if any(c.replace(' ', '').lower().startswith(p) for p in ('d', 'day', 't', 'v'))]
    if day_cols:
        def sort_key(c):
            try:
                return int(''.join(filter(str.isdigit, c)) or 0)
            except Exception:
                return 0
        day_cols.sort(key=sort_key)
        consumption_cols = day_cols
    else:
        consumption_cols = numeric

    return csv_path, label_col, consumption_cols


def fill_nan_linear_interpolation(X: np.ndarray, axis=1) -> np.ndarray:
    """
    Preenche NaN com interpolação linear ao longo do eixo temporal (Eq. 1 do artigo).
    X: shape (n_samples, n_days) ou (n_samples, n_days, ...); interpola ao longo axis.
    """
    X = np.asarray(X, dtype=np.float64)
    if not np.any(np.isnan(X)):
        return X.astype(np.float32)
    # Interpolar por linha (cada amostra é uma série temporal)
    if X.ndim == 2 and axis == 1:
        df = pd.DataFrame(X)
        out = df.interpolate(method="linear", axis=1, limit_direction="both").values
        return out.astype(np.float32)
    # Fallback: por linha, ao longo do eixo 1
    out = X.copy()
    for i in range(X.shape[0]):
        row = X[i].reshape(-1)
        nan_mask = np.isnan(row)
        if not nan_mask.any():
            continue
        valid = np.where(~nan_mask)[0]
        if len(valid) < 2:
            row[nan_mask] = np.nanmean(row)
            continue
        out_flat = np.interp(
            np.arange(len(row)),
            valid,
            row[valid],
        )
        out[i] = out_flat.reshape(X[i].shape)
    return out.astype(np.float32)


def ensure_1036_days(X_raw: np.ndarray, n_days: int) -> np.ndarray:
    """
    Garante 1036 dias por amostra. Se n_days == 1035, adiciona 1 dia (cópia do último).
    Retorna array (n, 1036).
    """
    n = X_raw.shape[0]
    if n_days >= NEEDED_DAYS:
        return X_raw[:, :NEEDED_DAYS].astype(np.float32)
    if n_days == ORIGINAL_DAYS:
        # 1035 → 1036: concatenar cópia do último dia
        last = X_raw[:, -1:]  # (n, 1)
        return np.concatenate([X_raw, last], axis=1).astype(np.float32)
    # Menos de 1035: repetir colunas até 1036
    repeated = (X_raw.T.repeat((NEEDED_DAYS // n_days) + 1, axis=0).T)[:, :NEEDED_DAYS]
    return repeated.astype(np.float32)


def load_sgcc_from_path(path, max_samples=None, seed=42, preprocessing="mean"):
    """
    Carrega o SGCC e retorna X (n, 148, 7, 1), y (n,) 0/1.

    preprocessing:
      - "mean": preenche NaN com média da coluna (comportamento legado).
      - "pereira": interpolação linear (Eq. 1) + 1035→1036 dias + reshape (148, 7).
    """
    csv_path, label_col, consumption_cols = _find_csv_and_label_column(path)
    if csv_path is None:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {path}")

    df = pd.read_csv(csv_path)
    if consumption_cols is None or len(consumption_cols) == 0:
        consumption_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if label_col and label_col in consumption_cols:
            consumption_cols.remove(label_col)
    n_days_avail = len(consumption_cols)
    if n_days_avail > NEEDED_DAYS:
        consumption_cols = consumption_cols[:NEEDED_DAYS]
        n_days_avail = NEEDED_DAYS
    elif n_days_avail < ORIGINAL_DAYS:
        # Menos de 1035: repetir colunas até 1036
        repeated = (consumption_cols * ((NEEDED_DAYS // n_days_avail) + 1))[:NEEDED_DAYS]
        consumption_cols = repeated
        n_days_avail = len(consumption_cols)
    # Se 1035 ou 1036, usa as colunas como estão (1035 será completado para 1036 depois)
    X_raw = df[consumption_cols].values.astype(np.float64)

    # Preenchimento de NaN
    if preprocessing == "pereira":
        X_raw = fill_nan_linear_interpolation(X_raw, axis=1)
        X_raw = ensure_1036_days(X_raw, n_days_avail)
    else:
        if np.any(np.isnan(X_raw)):
            col_mean = np.nanmean(X_raw, axis=0)
            inds = np.where(np.isnan(X_raw))
            X_raw[inds] = np.take(col_mean, inds[1])
        if n_days_avail < NEEDED_DAYS:
            X_raw = ensure_1036_days(X_raw, n_days_avail)
        else:
            X_raw = X_raw[:, :NEEDED_DAYS].astype(np.float32)

    if label_col and label_col in df.columns:
        y = df[label_col].values
        # Garantir binário 0/1
        y = (y > 0).astype(np.int32) if y.dtype in (float, np.float64, np.int64) else np.asarray(y, dtype=np.int32)
    else:
        # Sem label: gerar dummy para testar pipeline (todos 0)
        y = np.zeros(len(df), dtype=np.int32)

    if max_samples is not None:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X_raw))[:max_samples]
        X_raw = X_raw[idx]
        y = y[idx]

    # Reshape: (n, 1036) -> (n, 148, 7, 1) — 148 semanas × 7 dias
    n = X_raw.shape[0]
    if X_raw.shape[1] != NEEDED_DAYS:
        X_raw = ensure_1036_days(X_raw, X_raw.shape[1])
    X = X_raw.reshape(n, TIME_STEPS, FEATURES, 1).astype(np.float32)

    # Garantir zero NaN restantes (evita loss nan no treino; linhas totalmente NaN na interpolação)
    if np.any(np.isnan(X)):
        fill_val = np.nanmean(X)
        if np.isnan(fill_val):
            fill_val = 0.0
        X = np.nan_to_num(X, nan=fill_val, posinf=fill_val, neginf=fill_val)

    return X, y


def train_test_split(X, y, test_size=0.2, stratify=True, seed=42):
    """Split estratificado para treino e teste."""
    from sklearn.model_selection import train_test_split as sk_split
    if stratify and len(np.unique(y)) >= 2:
        X_train, X_test, y_train, y_test = sk_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )
    else:
        X_train, X_test, y_train, y_test = sk_split(
            X, y, test_size=test_size, random_state=seed
        )
    return X_train, X_test, y_train, y_test
