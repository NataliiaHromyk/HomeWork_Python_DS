from typing import Tuple, Dict, List, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def drop_unnecessary_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """
    Видаляє вказані колонки з DataFrame, якщо вони існують.
    """
    return df.drop(columns=[col for col in columns_to_drop if col in df.columns])


def preprocess_data(
    raw_df: pd.DataFrame,
    scale_numeric: bool = False
) -> Dict[str, object]:
    """
    Обробляє вхідний датафрейм: видаляє технічні колонки, кодує категоріальні,
    масштабує числові ознаки (опціонально) та ділить на train/val.

    :param raw_df: Початковий датафрейм
    :param scale_numeric: Чи застосовувати MinMaxScaler до числових колонок
    :return: Словник із train/val наборами, енкодером, скейлером і списками колонок
    """
    # Відкидаємо 'CustomerId' і 'Surname'
    df = drop_unnecessary_columns(raw_df, ['CustomerId', 'Surname'])

    # Цільова змінна
    target_col = 'Exited'

    # Train/val split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])
    input_cols = [col for col in df.columns if col != target_col]

    X_train = train_df[input_cols].copy()
    y_train = train_df[target_col].copy()
    X_val = val_df[input_cols].copy()
    y_val = val_df[target_col].copy()

    # Визначаємо типи колонок
    numeric_cols = X_train.select_dtypes(include='number').columns.tolist()
    categorical_cols = X_train.select_dtypes(include='object').columns.tolist()

    # One-hot encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(X_train[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols).tolist()

    X_train_encoded = pd.DataFrame(encoder.transform(X_train[categorical_cols]), columns=encoded_cols, index=X_train.index)
    X_val_encoded = pd.DataFrame(encoder.transform(X_val[categorical_cols]), columns=encoded_cols, index=X_val.index)

    X_train = pd.concat([X_train.drop(columns=categorical_cols), X_train_encoded], axis=1)
    X_val = pd.concat([X_val.drop(columns=categorical_cols), X_val_encoded], axis=1)

    # Масштабування
    if scale_numeric:
        scaler = MinMaxScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    else:
        scaler = None

    return {
        'train_X': X_train,
        'val_X': X_val,
        'train_y': y_train,
        'val_y': y_val,
        'encoder': encoder,
        'scaler': scaler,
        'input_cols': input_cols,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols
    }


def preprocess_new_data(
    new_df: pd.DataFrame,
    encoder: OneHotEncoder,
    scaler: Optional[MinMaxScaler],
    input_cols: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Обробляє новий датафрейм для передбачення:
    видаляє технічні поля, кодує, масштабує, повертає CustomerId + X.

    :param new_df: Нові вхідні дані (наприклад, з test.csv)
    :param encoder: Навчений OneHotEncoder
    :param scaler: Навчений MinMaxScaler або None
    :param input_cols: Список вхідних ознак
    :param numeric_cols: Числові ознаки
    :param categorical_cols: Категоріальні ознаки
    :return: (X_ready, customer_ids)
    """
    # Зберігаємо customer_id окремо
    customer_ids = new_df['CustomerId'].copy()

    # Відкидаємо непотрібне
    df = drop_unnecessary_columns(new_df.copy(), ['CustomerId', 'Surname'])
    X = df[input_cols].copy()

    # One-hot
    encoded_cols = encoder.get_feature_names_out(categorical_cols).tolist()
    encoded = pd.DataFrame(encoder.transform(X[categorical_cols]), columns=encoded_cols, index=X.index)
    X = pd.concat([X.drop(columns=categorical_cols), encoded], axis=1)

    # Масштабування
    if scaler is not None:
        X[numeric_cols] = scaler.transform(X[numeric_cols])

    return X, customer_ids
