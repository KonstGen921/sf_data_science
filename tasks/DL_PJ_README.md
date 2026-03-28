# DL_PJ — Бинарная классификация (Heart Disease)

Проект: анализ данных, EDA, обучение моделей и формирование submission для задачи бинарной классификации `class`.

## Содержание
- [1. Данные](#1-данные)
- [2. Первичный анализ](#2-первичный-анализ)
- [3. EDA (графики)](#3-eda-графики)
- [4. Обучение моделей](#4-обучение-моделей)
- [5. Предсказание и submission](#5-предсказание-и-submission)
- [6. Артефакты проекта](#6-артефакты-проекта)

---

## 1. Данные

### Размеры датасетов
| Датасет | Строк | Столбцов | Цель |
|---|---:|---:|---|
| train | 600 000 | 15 | `class` |
| test  | 400 000 | 14 | — |

**Постановка задачи:** целевая переменная `class` принимает значения 0/1 → задача **бинарной класс��фикации**.

---

## 2. Первичный анализ

### Типы признаков
- **Числовые (continuous):** `age`, `resting_blood_pressure`, `serum_cholestoral`, `maximum_heart_rate_achieved`, `oldpeak`
- **Категориальные/дискретные:** `sex`, `chest`, `fasting_blood_sugar`, `resting_electrocardiographic_results`,  
  `exercise_induced_angina`, `slope`, `number_of_major_vessels`, `thal`
- **`ID`** — идентификатор записи (не используется как признак при обучении; нужен для формирования submission)

### Качество данных (train): missing / unique / dtype
| feature | missing | unique | dtype |
|---|---:|---:|---|
| ID | 0 | 600000 | int64 |
| age | 0 | 594106 | float64 |
| sex | 0 | 2 | int64 |
| chest | 0 | 133009 | float64 |
| resting_blood_pressure | 0 | 596241 | float64 |
| serum_cholestoral | 0 | 598797 | float64 |
| fasting_blood_sugar | 0 | 2 | int64 |
| resting_electrocardiographic_results | 0 | 3 | int64 |
| maximum_heart_rate_achieved | 0 | 597583 | float64 |
| exercise_induced_angina | 0 | 2 | int64 |
| oldpeak | 0 | 384255 | float64 |
| slope | 0 | 3 | int64 |
| number_of_major_vessels | 0 | 4 | int64 |
| thal | 0 | 3 | int64 |
| class | 0 | 2 | int64 |

**Комментарий:** по смыслу `chest` должен быть категориальным (обычно 1–4), однако он имеет тип `float64` и очень много уникальных значений → признак требует предобработки (приведение к категориям и кодирование).

### Дубликаты и пропуски
- Пропусков: **0**
- Дубликатов: **0**

<details>
<summary>Показать df.info() (train/test)</summary>

```text
(сюда можно вставить вывод df.info(), если требуется)
```

</details>

---

## 3. EDA (графики)

### Выбросы (Boxplot)
Выбросы присутствуют в числовых признаках (особенно `oldpeak`), а также заметны хвосты у `serum_cholestoral` и `resting_blood_pressure`. При этом значения остаются в разумных диапазонах. Для базовой модели RandomForest выбросы отдельно не удалялись, т.к. модель устойчива к ним.

![Boxplot](/Users/konstantingeneralov/IDE/sf_data_science/tasks/1/boxplot_train.png)
![Boxplot](/Users/konstantingeneralov/IDE/sf_data_science/tasks/1/boxplot_test.png)

### Корреляционная матрица (числовые признаки)
- Наиболее заметная связь с `class`: `oldpeak` (+), `maximum_heart_rate_achieved` (-), `age` (+).
- Сильной мультиколлинеарности среди непрерывных признаков не наблюдается.

![Correlation heatmap](tasks/dl_pj_assets/corr_heatmap.png)

### Диаграммы рассеяния (Pairplot / Scatter)
Классы заметно перекрываются в большинстве пар признаков → простая линейная граница выражена слабо, поэтому целесообразны модели, учитывающие нелинейности и взаимодействия признаков.

![Pairplot](tasks/dl_pj_assets/pairplot.png)

---

## 4. Обучение моделей

Датасет `train` был разделён на train/validation для оценки качества.

### Почему ROC-AUC
ROC-AUC оценивает качество ранжирования: насколько модель в среднем присваивает объектам класса 1 более высокие вероятности, чем объектам класса 0. Метрика не привязана к одному порогу (например, 0.5), поэтому удобна для сравнения моделей.

### Результаты
| Модель | Особенности | ROC-AUC (val) |
|---|---|---:|
| Logistic Regression | стандартизация числовых признаков (`StandardScaler`) | 0.9557 |
| RandomForestClassifier | устойчив к выбросам, учитывает нелинейности | 0.9604 |

**Итоговая модель:** RandomForest (лучший ROC-AUC на валидации).

---

## 5. Предсказание и submission

Для тестовой выборки получены предсказания `class` и сформирован файл submission в формате:

- `ID` — идентификатор записи
- `class` — предсказанный класс (0/1)

Файл: `DL_PJ_submission_2.csv`.

---

## 6. Артефакты проекта
- Ноутбук: `tasks/DL_PJ.ipynb` / `tasks/DL_PJ copy.ipynb`
- Доп. описание: `tasks/DL_PJ_README.md`
- Изображения для README: `tasks/dl_pj_assets/*.png`