import argparse
import json
import joblib
import numpy as np
import pandas as pd
import torch

from model import MLP  # импортируем класс модели MLP из файла model.py, который мы реализовали ранее


def sigmoid(x: np.ndarray) -> np.ndarray:    # функция для преобразования логитов в вероятности с помощью сигмоиды, которая используется для бинарной классификации
    return 1.0 / (1.0 + np.exp(-x))


def main():   # основная функция, которая выполняет весь процесс инференса: загрузка данных, предобработка, загрузка модели, предсказание и сохранение результатов
    parser = argparse.ArgumentParser() # создаем парсер аргументов командной строки для удобного запуска скрипта с разными параметрами
    parser.add_argument("--input", required=True, help="Path to test CSV (e.g. DL_PJ_test.csv).") 
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path.")
    parser.add_argument("--ckpt", default="models/mlp_best.ckpt", help="Path to model checkpoint.")
    parser.add_argument("--scaler", default="artifacts/scaler.pkl", help="Path to scaler.pkl")
    parser.add_argument("--encoder", default="artifacts/encoder.pkl", help="Path to encoder.pkl")
    parser.add_argument("--feature-cols", default="artifacts/feature_cols.json", help="Path to feature_cols.json")
    parser.add_argument("--id-col", default="ID", help="ID column name in test CSV.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"], help="Device to use.")
    args = parser.parse_args()

    # загрузка данных
    df = pd.read_csv(args.input) # загружаем тестовый CSV файл в DataFrame для дальнейшей обработки и предсказания

    # загрузка предобработчиков: scaler для числовых признаков и encoder для категориальных признаков, а также список всех признаков в правильном порядке, который использовался при обучении модели
    scaler = joblib.load(args.scaler) # загружаем scaler из файла 
    encoder = joblib.load(args.encoder) # загружаем encoder из файла
    with open(args.feature_cols, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    # числовые и категориальные признаки: мы знаем, что числовые признаки - это первые 5 столбцов в списке feature_cols, а категориальные признаки - это исходные столбцы, 
    # которые были использованы для создания OHE в нашем ноутбуке, и которые мы должны обработать с помощью encoder
    
    num_colls = feature_cols[:5] 

    
    cat_coll = [
        "sex",
        "chest",
        "fasting_blood_sugar",
        "resting_electrocardiographic_results",
        "exercise_induced_angina",
        "slope",
        "number_of_major_vessels",
        "thal",
    ]

    # обработка числовых признаков: сначала преобразуем их в float32 для экономии памяти, затем применяем scaler для стандартизации, и снова приводим к float32 для совместимости с моделью
    x_num = df[num_colls].astype(np.float32)
    x_num_scaled = scaler.transform(x_num.to_numpy()).astype(np.float32)

    # обработка категориальных признаков: применяем encoder для получения OHE представления, и приводим результат к float32 для совместимости с моделью
    x_cat_ohe = encoder.transform(df[cat_coll])
    x_cat_ohe = np.asarray(x_cat_ohe, dtype=np.float32)

    # соединяем числовые и категориальные признаки в один массив для подачи на вход модели, и приводим к float32 для совместимости с моделью
    X_np = np.concatenate([x_num_scaled, x_cat_ohe], axis=1).astype(np.float32)

    # создаем DataFrame из объединенных признаков, чтобы гарантировать правильный порядок столбцов, который соответствует тому, что использовался при обучении модели, 
    # и заполняем отсутствующие столбцы нулями, если какие-то признаки отсутствуют в тестовом наборе данных
    # это гарантирует, что порядок признаков в X_df соответствует тому, что ожидала модель при обучении
    ohe_cols = list(encoder.get_feature_names_out(cat_coll))
    X_df = pd.DataFrame(
        np.concatenate([x_num_scaled, x_cat_ohe], axis=1),
        columns=list(num_colls) + ohe_cols,
        index=df.index,
    )
    X_df = X_df.reindex(columns=feature_cols, fill_value=0.0)
    X_np = X_df.to_numpy().astype(np.float32)

    # определяем устройство для инференса: если указано "auto", то выбираем MPS (Metal Performance Shaders) для Mac с поддержкой MPS, иначе используем CPU, или конкретное устройство, указанное в аргументах
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # загрузка модели из контрольной точки, которая была сохранена во время обучения, и перевод модели на выбранное устройство для инференса
    model = MLP.load_from_checkpoint(args.ckpt).to(device)
    model.eval()

    # предсказание логитов модели на тестовых данных: мы отключаем градиенты с помощью torch.no_grad(), передаем данные в виде тензора на устройство, 
    # получаем логиты, и преобразуем их в вероятности с помощью сигмоиды
    with torch.no_grad():
        logits = model(torch.from_numpy(X_np).to(device)).cpu().numpy().reshape(-1)
        proba = sigmoid(logits)

    out = pd.DataFrame({args.id_col: df[args.id_col].values, "proba": proba})
    out.to_csv(args.output, index=False)

    print(f"Saved {args.output} with shape {out.shape} on device={device}")


if __name__ == "__main__":
    main()