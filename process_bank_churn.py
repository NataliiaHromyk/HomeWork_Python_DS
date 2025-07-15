from typing import Tuple, Dict, List, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def drop_unnecessary_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """
    Видаляє вказані колонки з DataFrame, якщо вони існують.

    :param df: Вхідний DataFrame
    :param columns_to_drop: Список назв колонок для видалення
    :return: Оновлений DataFrame
    """
    return df.drop(columns=[col for col in columns_to_drop if col in df.columns])


def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Розділяє DataFrame на train і validation множини.

    :param df: Оброблений DataFrame
    :param target_col: Назва цільової колонки
    :param test_size: Частка даних для валідації
    :param random_state: Початкове значення для генератора
    :return: X_train, X_val, y_train, y_val
    """
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target_col])
    input_cols = [col for col in df.columns if col != target_col]
    return train_df[input_cols], val_df[input_cols], train_df[target_col], val_df[target_col]


def encode_categorical_features(
    X_train: pd.DataFrame, 
    X_val: pd.DataFrame, 
    categorical_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, List[str]]:
    """
    Застосовує OneHotEncoding до категоріальних колонок.

    :param X_train: Тренувальні ознаки
    :param X_val: Валідаційні ознаки
    :param categorical_cols: Назви категоріальних колонок
    :return: Закодовані X_train, X_val, енкодер, список нових колонок
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(X_train[categorical_cols])
    encoder_cols = list(encoder.get_feature_names_out(categorical_cols))

    X_train_encoded = pd.DataFrame(encoder.transform(X_train[categorical_cols]), columns=encoder_cols, index=X_train.index)
    X_val_encoded = pd.DataFrame(encoder.transform(X_val[categorical_cols]), columns=encoder_cols, index=X_val.index)

    X_train = pd.concat([X_train.drop(columns=categorical_cols), X_train_encoded], axis=1)
    X_val = pd.concat([X_val.drop(columns=categorical_cols), X_val_encoded], axis=1)

    return X_train, X_val, encoder, encoder_cols


def scale_numeric_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    numeric_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Масштабує числові ознаки за допомогою MinMaxScaler.

    :param X_train: Тренувальні ознаки
    :param X_val: Валідаційні ознаки
    :param numeric_cols: Назви числових колонок
    :return: Масштабовані X_train, X_val, скейлер
    """
    scaler = MinMaxScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    return X_train, X_val, scaler


def preprocess_data(
    raw_df: pd.DataFrame, 
    scale_numeric: bool = False
) -> Dict[str, object]:
    """
    Основна функція для обробки даних. Включає: видалення колонок, енкодинг, масштабування.

    :param raw_df: Вхідний датафрейм
    :param scale_numeric: Чи масштабувати числові ознаки
    :return: Словник з обробленими даними, енкодером, скейлером і колонками
    """
    df = drop_unnecessary_columns(raw_df, ['Surname'])
    X_train, X_val, y_train, y_val = split_data(df, target_col='Exited')

    numeric_cols = X_train.select_dtypes(include='number').columns.tolist()
    categorical_cols = X_train.select_dtypes(include='object').columns.tolist()

    X_train, X_val, encoder, encoder_cols = encode_categorical_features(X_train, X_val, categorical_cols)

    if scale_numeric:
        X_train, X_val, scaler = scale_numeric_features(X_train, X_val, numeric_cols)
    else:
        scaler = None

    return {
        'train_X': X_train,
        'val_X': X_val,
        'train_y': y_train,
        'val_y': y_val,
        'encoder': encoder,
        'scaler': scaler,
        'input_cols': X_train.columns.tolist(),
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
) -> pd.DataFrame:
    """
    Обробляє нові дані для передбачення, використовуючи навчені скейлер та енкодер.

    :param new_df: Новий датафрейм (наприклад, з test.csv)
    :param encoder: Навчений OneHotEncoder
    :param scaler: Навчений MinMaxScaler або None
    :param input_cols: Всі колонки, що використовувались у моделі
    :param numeric_cols: Числові колонки
    :param categorical_cols: Категоріальні колонки
    :return: Оброблений датафрейм
    """
    df = drop_unnecessary_columns(new_df.copy(), ['Surname'])
    X = df[input_cols].copy()

    # OneHotEncoding
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    encoded = pd.DataFrame(encoder.transform(X[categorical_cols]), columns=encoded_cols, index=X.index)
    X = pd.concat([X.drop(columns=categorical_cols), encoded], axis=1)

    # Масштабування
    if scaler is not None:
        X[numeric_cols] = scaler.transform(X[numeric_cols])

    return X