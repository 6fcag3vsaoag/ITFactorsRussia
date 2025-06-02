import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from ipywidgets import interact, FloatSlider, Dropdown, fixed
from IPython.display import display

# Словарь файлов и соответствующих им названий столбцов
files = {
    'tksg.csv': 'TCSG',
    'yandex.csv': 'YDEX',
    'vk.csv': 'VKCO',
    'rostelecom.csv': 'RTKM',
    'usd_rub.csv': 'USD_RUB',
    'brent.csv': 'Brent',
    'bitcoin.csv': 'BTC_USD'
}

# Чтение и отображение первых и последних 5 строк каждого файла и статистики
for file_name, column_name in files.items():
    try:
        df = pd.read_csv(file_name)
        print(f"\nСтатистика файла {file_name}:")
        print(f"Количество строк: {len(df)}")
        print(f"Количество столбцов: {len(df.columns)}")
        
        print("\nПервые 5 строк файла:")
        print(df.head())
        print("\nПоследние 5 строк файла:")
        print(df.tail())
        print("-" * 80)
    except FileNotFoundError:
        print(f"Файл {file_name} не найден")
    except Exception as e:
        print(f"Ошибка при чтении файла {file_name}: {str(e)}")
        
        
# 1 обьединение данных в единый временной ряд методом inner join

# Функция для конвертации строк с числами в float
def convert_to_float(value):
    try:
        # Удаляем точку как разделитель тысяч и заменяем запятую на точку для десятичного разделителя
        cleaned_value = str(value).replace('.', '').replace(',', '.')
        return float(cleaned_value)
    except (ValueError, TypeError):
        return value  # Возвращаем исходное значение, если конвертация не удалась

# Создание единого датафрейма final_df
dfs = []
for file_name, column_name in files.items():
    try:
        df = pd.read_csv(file_name)
        # Выбираем только нужные столбцы и переименовываем столбец с ценой
        df = df[['Дата', 'Цена']].rename(columns={'Цена': column_name})
        dfs.append(df)
    except FileNotFoundError:
        print(f"Файл {file_name} не найден")
    except Exception as e:
        print(f"Ошибка при чтении файла {file_name}: {str(e)}")

# Объединение всех датафреймов
if dfs:
    # Начинаем с первого датафрейма
    final_df = dfs[0]
    # Последовательно объединяем с остальными
    for df in dfs[1:]:
        final_df = pd.merge(final_df, df, on='Дата', how='inner')
    
    # Сортировка по дате
    final_df['Дата'] = pd.to_datetime(final_df['Дата'], format='%d.%m.%Y')
    final_df = final_df.sort_values('Дата')
    
    # Преобразование данных в числовой формат
    for column in final_df.columns:
        if column != 'Дата':
            final_df[column] = final_df[column].apply(convert_to_float)
    
    # Вывод информации и статистики до очистки
    print("\nИтоговый датафрейм:")
    print("\nПервые 5 строк:")
    print(final_df.head())
    print("\nПоследние 5 строк:")
    print(final_df.tail())
    print("\nИнформация о датафрейме:")
    print(final_df.info())
    print("\nСтатистика датафрейма:")
    print(final_df.describe())
    
    # Построение графика
    plt.figure(figsize=(15, 8))
    
    # Нормализация данных (приведение к базе 100)
    normalized_df = final_df.copy()
    for column in normalized_df.columns:
        if column != 'Дата':
            normalized_df[column] = normalized_df[column] / normalized_df[column].iloc[0] * 100
    
    # Построение графика для каждого актива
    for column in normalized_df.columns:
        if column != 'Дата':
            plt.plot(normalized_df['Дата'], normalized_df[column], label=column)
    
    plt.title('Динамика активов (база 100)')
    plt.xlabel('Дата')
    plt.ylabel('Значение (база 100)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Не удалось создать датафрейм из-за ошибок при чтении файлов")
    

# Очистка от выбросов (менее строгая: удаляем строки, где ВСЕ столбцы являются выбросами)
def remove_outliers(df, columns):
    mask = pd.Series(True, index=df.index)  # Изначально все строки включены
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Обновляем маску: строка остается, если значение в столбце НЕ является выбросом
        mask = mask & (df[col].between(lower_bound, upper_bound))
    return df[mask]

# Очистка от выбросов
columns_to_clean = list(files.values())
cleaned_df = remove_outliers(final_df, columns_to_clean)

# Статистика после очистки
print("\nИтоговый датафрейм после очистки от выбросов:")
print("\nПервые 5 строк:")
print(cleaned_df.head())
print("\nПоследние 5 строк:")
print(cleaned_df.tail())
print("\nСтатистика после очистки:")
print(cleaned_df.info())
# print(cleaned_df.describe())

# Построение графика для очищенных данных
plt.figure(figsize=(15, 8))

# Нормализация данных (приведение к базе 100) для очищенного датафрейма
normalized_df = cleaned_df.copy()
for column in normalized_df.columns:
    if column != 'Дата':
        normalized_df[column] = normalized_df[column] / normalized_df[column].iloc[0] * 100

# Построение графика для каждого актива
for column in normalized_df.columns:
    if column != 'Дата':
        plt.plot(normalized_df['Дата'], normalized_df[column], label=column)

plt.title('Динамика активов (база 100, после очистки от выбросов)')
plt.xlabel('Дата')
plt.ylabel('Значение (база 100)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 3. Нормализация (Z-нормализация)
# normalized_df = final_df.copy()
# for col in columns_to_clean:
#     mean = final_df[col].mean()
#     std = final_df[col].std()
#     normalized_df[col] = (final_df[col] - mean) / std
#     print(f"\nСтатистика для {col} (до нормализации): mean = {mean:.2f}, std = {std:.2f}")


normalized_df = cleaned_df.copy()
for col in columns_to_clean:
    mean = cleaned_df[col].mean()
    std = cleaned_df[col].std()
    normalized_df[col] = (cleaned_df[col] - mean) / std
    print(f"\nСтатистика для {col} (до нормализации): mean = {mean:.2f}, std = {std:.2f}")

print("\nПример нормализованного датафрейма (первые 5 строк):")
print(normalized_df.head())
print("\nСтатистика после нормализации:")
print(normalized_df.describe())


# 1. Проверка нормальности и характеристик распределения
factors = ['TCSG', 'YDEX', 'VKCO', 'RTKM', 'USD_RUB', 'Brent', 'BTC_USD']
normality_results = []
distribution_stats = []

for factor in factors:
    data = normalized_df[factor].dropna()
    # Тест Шапиро-Уилка
    stat, p_value = stats.shapiro(data)
    conclusion = "Не соответствует нормальному распределению" if p_value < 0.05 else "Соответствует нормальному распределению"
    normality_results.append([factor, stat, p_value, conclusion])

    # Характеристики распределения
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    mean = data.mean()
    median = data.median()
    std = data.std()
    min_val = data.min()
    max_val = data.max()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    
    # Интерпретация асимметрии
    skewness_comment = (
        "Сильная правосторонняя асимметрия" if skewness > 1 else
        "Сильная левосторонняя асимметрия" if skewness < -1 else
        "Умеренная асимметрия" if abs(skewness) > 0.5 else
        "Приблизительно симметричное распределение"
    )
    
    # Интерпретация куртозиса
    kurtosis_comment = (
        "Тяжёлые хвосты (много выбросов)" if kurtosis > 3 else
        "Лёгкие хвосты (меньшая вероятность экстремальных значений, что положительно для моделирования)" if kurtosis < -1 else
        "Нормальные хвосты (умеренное количество выбросов)"
    )
    
    # Дополнительные комментарии для факторов
    factor_comment = (
        "Высокая волатильность из-за влияния криптовалютного рынка и санкций." if factor == 'TCSG' else
        "Зависимость от сырьевых рынков (нефть) и макроэкономической среды." if factor in ['YDEX', 'VKCO', 'RTKM'] else
        "Волатильность связана с экономическими и геополитическими факторами." if factor == 'USD_RUB' else
        "Зависимость от мировых сырьевых рынков." if factor == 'Brent' else
        "Высокая волатильность из-за спекулятивного характера криптовалют."
    )
    
    distribution_stats.append([
        factor, mean, median, std, skewness, kurtosis, min_val, max_val, q1, q3,
        skewness_comment, kurtosis_comment, factor_comment
    ])

# Формирование таблиц результатов
normality_df = pd.DataFrame(normality_results, columns=['Фактор', 'Статистика Шапиро-Уилка', 'p-value', 'Нормальность'])
dist_stats_df = pd.DataFrame(distribution_stats, columns=[
    'Фактор', 'Среднее', 'Медиана', 'Стд. откл.', 'Асимметрия', 'Куртозис',
    'Мин.', 'Макс.', 'Q1', 'Q3', 'Асимметрия (пояснение)', 'Куртозис (пояснение)', 'Комментарий'
]).round(4)

# Вывод таблиц
print("\nРезультаты теста Шапиро-Уилка:")
print(normality_df.to_string(index=False))
# print("\nХарактеристики распределения:")
# print(dist_stats_df.to_string(index=False))


# 3. Корреляционный анализ 
def analyze_correlations():
    print("\n=== Корреляционный анализ (метод Спирмена) ===")
    
    # Корреляция Спирмена
    correlation_matrix = normalized_df[columns_to_clean].corr(method='spearman')
    
    # # Форматирование матрицы для вывода
    # corr_df = correlation_matrix.round(3)  # Округляем до 3 знаков после запятой
    # print("\nМатрица корреляции (Спирмен):")
    # display(corr_df.style.background_gradient(cmap='coolwarm', axis=None).set_caption('Корреляционная матрица (Спирмен)'))

    # Визуализация корреляции
    plt.figure(figsize=(10, 8), dpi=100)
    plt.matshow(correlation_matrix, cmap='coolwarm', fignum=1)
    plt.xticks(range(len(columns_to_clean)), columns_to_clean, rotation=45, ha='left')
    plt.yticks(range(len(columns_to_clean)), columns_to_clean)
    for (i, j), val in np.ndenumerate(correlation_matrix):
        plt.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=10)
    plt.title('Корреляция Спирмена', fontsize=14, pad=20)
    plt.colorbar(label='Коэффициент корреляции')
    plt.tight_layout()
    plt.show()

# Выполняем корреляционный анализ
analyze_correlations()


# 2. Модель линейной регрессии
dependent_vars = ['TCSG', 'YDEX', 'VKCO', 'RTKM']
independent_vars = ['USD_RUB', 'Brent', 'BTC_USD']

results_normalized = {}  # Линейная регрессия
X_normalized = normalized_df[independent_vars].dropna()

for y_var in dependent_vars:
    y = normalized_df[y_var].dropna()
    common_index = X_normalized.index.intersection(y.index)
    X_subset = X_normalized.loc[common_index]
    y_subset = y.loc[common_index]
    
    # Разбиение данных
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)
    
    # Линейная регрессия
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    y_pred_lr = model_lr.predict(X_test)
    mae_lr = np.mean(np.abs(y_test - y_pred_lr))
    mse_lr = np.mean((y_test - y_pred_lr) ** 2)
    
    results_normalized[y_var] = {
        'y_actual': y_test,
        'y_pred': y_pred_lr,
        'X_test': X_test,
        'model': model_lr,
        'coefficients': dict(zip(independent_vars, model_lr.coef_)),
        'intercept': model_lr.intercept_,
        'MAE': mae_lr,
        'MSE': mse_lr
    }
    
# Формирование таблицы регрессии
results_data = []
for y_var in dependent_vars:
    coeffs = results_normalized[y_var]['coefficients']
    results_data.append([
        y_var,
        coeffs.get('USD_RUB', 0),
        coeffs.get('Brent', 0),
        coeffs.get('BTC_USD', 0),
        results_normalized[y_var]['intercept'],
        results_normalized[y_var]['MAE'],
        results_normalized[y_var]['MSE']
    ])
results_df = pd.DataFrame(
    results_data,
    columns=['Компания', 'USD_RUB (β1)', 'Brent (β2)', 'BTC_USD (β3)', 'Intercept', 'MAE', 'MSE']
).round(4)

print("\nРезультаты линейной регрессии (на тестовой выборке):")
print(results_df.to_string(index=False))

# Анализ ошибок регрессии
print("\nАнализ ошибок линейной регрессии (на тестовой выборке):")
for y_var in dependent_vars:
    errors = results_normalized[y_var]['y_actual'] - results_normalized[y_var]['y_pred']
    error_mean = errors.mean()
    error_std = errors.std()
    error_skewness = stats.skew(errors)
    error_kurtosis = stats.kurtosis(errors)
    
    stat, p = stats.shapiro(errors)
    normality_comment = "Ошибки не соответствуют нормальному распределению" if p < 0.05 else "Ошибки соответствуют нормальному распределению"
    
    error_mean_comment = (
        f"Ошибки в среднем {'смещены вверх' if error_mean > 0.1 else 'смещены вниз' if error_mean < -0.1 else 'близки к нулю'} "
        f"(среднее: {error_mean:.4f})."
    )
    error_std_comment = (
        f"{'Высокая вариабельность ошибок' if error_std > 1 else 'Умеренная вариабельность ошибок' if error_std > 0.5 else 'Низкая вариабельность ошибок'} "
        f"(стандартное отклонение: {error_std:.4f})."
    )
    error_skewness_comment = (
        "Сильная правосторонняя асимметрия ошибок" if error_skewness > 1 else
        "Сильная левосторонняя асимметрия ошибок" if error_skewness < -1 else
        "Умеренная асимметрия ошибок" if abs(error_skewness) > 0.5 else
        "Ошибки симметричны"
    )
    error_kurtosis_comment = (
        "Тяжёлые хвосты ошибок (много выбросов)" if error_kurtosis > 3 else
        "Лёгкие хвосты ошибок (меньшая вероятность экстремальных значений)" if error_kurtosis < -1 else
        "Нормальные хвосты ошибок"
    )
    
    # print(f"\nХарактеристики ошибок для {y_var}:")
    # print(f"- Среднее: {error_mean_comment}")
    # print(f"- Стандартное отклонение: {error_std_comment}")
    # print(f"- Асимметрия: {error_skewness_comment} (значение: {error_skewness:.4f})")
    # print(f"- Куртозис: {error_kurtosis_comment} (значение: {error_kurtosis:.4f})")
    # print(f"- Шапиро-Уилка: stat={stat:.4f}, p={p:.4f}, {normality_comment}")
    
    # Гистограмма ошибок
    plt.figure(figsize=(8, 6))
    sns.histplot(errors, kde=True, bins=50, color='purple')
    plt.title(f'Распределение ошибок для {y_var} (линейная регрессия)')
    plt.xlabel('Ошибка (реальное - предсказанное)')
    plt.ylabel('Частота')
    plt.xlim(-5, 5)
    plt.grid(True)
    plt.show()
    
    # График предсказанных vs реальных значений
    y_actual = results_normalized[y_var]['y_actual']
    y_pred = results_normalized[y_var]['y_pred']
    plt.figure(figsize=(8, 6))
    plt.scatter(y_actual, y_pred, color='blue', alpha=0.5)
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
    plt.title(f'Предсказанные vs Реальные значения для {y_var} (линейная регрессия)')
    plt.xlabel('Реальные значения (Z-нормализованные)')
    plt.ylabel('Предсказанные значения (Z-нормализованные)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
# 3. Модель случайного леса
dependent_vars = ['TCSG', 'YDEX', 'VKCO', 'RTKM']
independent_vars = ['USD_RUB', 'Brent', 'BTC_USD']

results_normalized_rf = {}  # Случайный лес
X_normalized = normalized_df[independent_vars].dropna()

for y_var in dependent_vars:
    y = normalized_df[y_var].dropna()
    common_index = X_normalized.index.intersection(y.index)
    X_subset = X_normalized.loc[common_index]
    y_subset = y.loc[common_index]
    
    # Разбиение данных
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)
    
    # Случайный лес
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)
    mae_rf = np.mean(np.abs(y_test - y_pred_rf))
    mse_rf = np.mean((y_test - y_pred_rf) ** 2)
    
    results_normalized_rf[y_var] = {
        'y_actual': y_test,
        'y_pred': y_pred_rf,
        'X_test': X_test,
        'model': model_rf,
        'feature_importances': dict(zip(independent_vars, model_rf.feature_importances_)),
        'MAE': mae_rf,
        'MSE': mse_rf
    }
    
# Формирование таблицы для случайного леса
results_data_rf = []
for y_var in dependent_vars:
    importances = results_normalized_rf[y_var]['feature_importances']
    results_data_rf.append([
        y_var,
        importances.get('USD_RUB', 0),
        importances.get('Brent', 0),
        importances.get('BTC_USD', 0),
        results_normalized_rf[y_var]['MAE'],
        results_normalized_rf[y_var]['MSE']
    ])
results_df_rf = pd.DataFrame(
    results_data_rf,
    columns=['Компания', 'USD_RUB (важность)', 'Brent (важность)', 'BTC_USD (важность)', 'MAE', 'MSE']
).round(4)

print("\nРезультаты случайного леса (на тестовой выборке):")
print(results_df_rf.to_string(index=False))

# Анализ ошибок случайного леса
print("\nАнализ ошибок случайного леса (на тестовой выборке):")
for y_var in dependent_vars:
    errors = results_normalized_rf[y_var]['y_actual'] - results_normalized_rf[y_var]['y_pred']
    error_mean = errors.mean()
    error_std = errors.std()
    error_skewness = stats.skew(errors)
    error_kurtosis = stats.kurtosis(errors)
    
    stat, p = stats.shapiro(errors)
    normality_comment = "Ошибки не соответствуют нормальному распределению" if p < 0.05 else "Ошибки соответствуют нормальному распределению"
    
    error_mean_comment = (
        f"Ошибки в среднем {'смещены вверх' if error_mean > 0.1 else 'смещены вниз' if error_mean < -0.1 else 'близки к нулю'} "
        f"(среднее: {error_mean:.4f})."
    )
    error_std_comment = (
        f"{'Высокая вариабельность ошибок' if error_std > 1 else 'Умеренная вариабельность ошибок' if error_std > 0.5 else 'Низкая вариабельность ошибок'} "
        f"(стандартное отклонение: {error_std:.4f})."
    )
    error_skewness_comment = (
        "Сильная правосторонняя асимметрия ошибок" if error_skewness > 1 else
        "Сильная левосторонняя асимметрия ошибок" if error_skewness < -1 else
        "Умеренная асимметрия ошибок" if abs(error_skewness) > 0.5 else
        "Ошибки симметричны"
    )
    error_kurtosis_comment = (
        "Тяжёлые хвосты ошибок (много выбросов)" if error_kurtosis > 3 else
        "Лёгкие хвосты ошибок (меньшая вероятность экстремальных значений)" if error_kurtosis < -1 else
        "Нормальные хвосты ошибок"
    )
    
    # print(f"\nХарактеристики ошибок для {y_var}:")
    # print(f"- Среднее: {error_mean_comment}")
    # print(f"- Стандартное отклонение: {error_std_comment}")
    # print(f"- Асимметрия: {error_skewness_comment} (значение: {error_skewness:.4f})")
    # print(f"- Куртозис: {error_kurtosis_comment} (значение: {error_kurtosis:.4f})")
    # print(f"- Шапиро-Уилка: stat={stat:.4f}, p={p:.4f}, {normality_comment}")
    
    # # Гистограмма ошибок
    # plt.figure(figsize=(8, 6))
    # sns.histplot(errors, kde=True, bins=50, color='purple')
    # plt.title(f'Распределение ошибок для {y_var} (случайный лес)')
    # plt.xlabel('Ошибка (реальное - предсказанное)')
    # plt.ylabel('Частота')
    # plt.xlim(-5, 5)
    # plt.grid(True)
    # plt.show()
    
    # # График предсказанных vs реальных значений
    # y_actual = results_normalized_rf[y_var]['y_actual']
    # y_pred = results_normalized_rf[y_var]['y_pred']
    # plt.figure(figsize=(8, 6))
    # plt.scatter(y_actual, y_pred, color='blue', alpha=0.5)
    # plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
    # plt.title(f'Предсказанные vs Реальные значения для {y_var} (случайный лес)')
    # plt.xlabel('Реальные значения (Z-нормализованные)')
    # plt.ylabel('Предсказанные значения (Z-нормализованные)')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    
# 4. ANOVA анализ (только для линейной регрессии)
print("\nВывод ANOVA для линейной регрессии:")
results_anova = {}
for y_var in dependent_vars:
    data = normalized_df[[y_var] + independent_vars].dropna()
    formula = f'{y_var} ~ USD_RUB + Brent + BTC_USD'
    model = ols(formula, data=data).fit()
    anova_table = anova_lm(model, typ=2)
    results_anova[y_var] = anova_table

    print(f"\nРезультаты ANOVA для {y_var}:")
    print(results_anova[y_var].to_string(index=False))
    
# 6. Интерактивная визуализация

def train_ruble_model():
    X_ruble = cleaned_df[independent_vars].dropna()
    results_ruble = {}
    results_ruble_rf = {}
    
    for y_var in dependent_vars:
        y = cleaned_df[y_var].dropna()
        common_index = X_ruble.index.intersection(y.index)
        X_subset = X_ruble.loc[common_index]
        y_subset = y.loc[common_index]
        
        X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)
        
        # Линейная регрессия
        model_lr = LinearRegression()
        model_lr.fit(X_train, y_train)
        y_pred_lr = model_lr.predict(X_test)
        mae_lr = np.mean(np.abs(y_test - y_pred_lr))
        mse_lr = np.mean((y_test - y_pred_lr) ** 2)
        
        results_ruble[y_var] = {
            'y_actual': y_test,
            'y_pred': y_pred_lr,
            'X_test': X_test,
            'model': model_lr,  # Сохраняем модель
            'coefficients': dict(zip(independent_vars, model_lr.coef_)),
            'intercept': model_lr.intercept_,
            'MAE': mae_lr,
            'MSE': mse_lr
        }
        
        # Случайный лес
        model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        model_rf.fit(X_train, y_train)
        y_pred_rf = model_rf.predict(X_test)
        mae_rf = np.mean(np.abs(y_test - y_pred_rf))
        mse_rf = np.mean((y_test - y_pred_rf) ** 2)
        
        results_ruble_rf[y_var] = {
            'y_actual': y_test,
            'y_pred': y_pred_rf,
            'X_test': X_test,
            'model': model_rf,  # Сохраняем модель
            'feature_importances': dict(zip(independent_vars, model_rf.feature_importances_)),
            'MAE': mae_rf,
            'MSE': mse_rf
        }
    
    return results_ruble, results_ruble_rf

results_ruble, results_ruble_rf = train_ruble_model()

def predict_prices(usd_rub, brent, btc_usd, data_type, model_type):
    predictions = {}
    if data_type == 'Z-нормализованные':
        results = results_normalized if model_type == 'Линейная регрессия' else results_normalized_rf
        usd_rub_norm = (usd_rub - cleaned_df['USD_RUB'].mean()) / cleaned_df['USD_RUB'].std()
        brent_norm = (brent - cleaned_df['Brent'].mean()) / cleaned_df['Brent'].std()
        btc_usd_norm = (btc_usd - cleaned_df['BTC_USD'].mean()) / cleaned_df['BTC_USD'].std()
        ylabel = 'Цена акций (Z-нормализованная)'
    else:
        results = results_ruble if model_type == 'Линейная регрессия' else results_ruble_rf
        usd_rub_norm = usd_rub
        brent_norm = brent
        btc_usd_norm = btc_usd
        ylabel = 'Цена акций (рубли)'
    
    # Создаем DataFrame для предсказания
    input_data = pd.DataFrame({
        'USD_RUB': [usd_rub_norm],
        'Brent': [brent_norm],
        'BTC_USD': [btc_usd_norm]
    })
    
    for y_var in dependent_vars:
        if model_type == 'Линейная регрессия':
            model_coefs = results[y_var]['coefficients']
            intercept = results[y_var]['intercept']
            prediction = (intercept +
                          model_coefs['USD_RUB'] * usd_rub_norm +
                          model_coefs['Brent'] * brent_norm +
                          model_coefs['BTC_USD'] * btc_usd_norm)
        else:
            # Используем сохраненную модель
            model = results[y_var]['model']
            prediction = model.predict(input_data)[0]
        
        if data_type == 'Реальные (рубли)':
            prediction = max(0, prediction)
        predictions[y_var] = prediction
    return predictions, ylabel

def update_plot(usd_rub, brent, btc_usd, data_type, model_type):
    try:
        predictions, ylabel = predict_prices(usd_rub, brent, btc_usd, data_type, model_type)
        print(f"\nпрогнозы ({data_type}, {model_type}):\n", {k: round(v, 2) for k, v in predictions.items()})
        
        # Барплот предсказаний
        plt.figure(figsize=(8, 6))
        companies = list(predictions.keys())
        values = [predictions[company] for company in companies]
        bars = plt.bar(companies, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.title(f'Прогнозируемые цены акций ({data_type}, {model_type})', fontsize=14)
        plt.xlabel('Компания')
        plt.ylabel(ylabel)
        plt.grid(True, axis='y')
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, value + 0.02 * value, f'{value:.2f}', ha='center', va='bottom')
        plt.tight_layout()
        # plt.savefig(f'prediction_{data_type.lower().replace(" ", "_")}_{model_type.lower().replace(" ", "_")}_barplot.png', bbox_inches='tight')
        plt.show()

        # Отладочный вывод
        results = (
            results_normalized if data_type == 'Z-нормализованные' and model_type == 'Линейная регрессия' else
            results_normalized_rf if data_type == 'Z-нормализованные' and model_type == 'Случайный лес' else
            results_ruble if data_type == 'Реальные (рубли)' and model_type == 'Линейная регрессия' else
            results_ruble_rf
        )
        print(f"\nРезультаты модели ({data_type}, {model_type}):")
        for y_var in dependent_vars:
            print(f"{y_var}:")
            if model_type == 'Линейная регрессия':
                print(f"  USD_RUB: {results[y_var]['coefficients']['USD_RUB']:.2f}")
                print(f"  Brent: {results[y_var]['coefficients']['Brent']:.2f}")
                print(f"  BTC_USD: {results[y_var]['coefficients']['BTC_USD']:.2f}")
                print(f"  Intercept: {results[y_var]['intercept']:.2f}")
            else:
                print(f"  USD_RUB importance: {results[y_var]['feature_importances']['USD_RUB']:.2f}")
                print(f"  Brent importance: {results[y_var]['feature_importances']['Brent']:.2f}")
                print(f"  BTC_USD importance: {results[y_var]['feature_importances']['BTC_USD']:.2f}")
            print(f"  MAE: {results[y_var]['MAE']:.4f}")
            print(f"  MSE: {results[y_var]['MSE']:.4f}")

    except Exception as e:
        print(f"Ошибка в интерактивной визуализации: {e}")

# Ползунки для реальных данных
usd_rub_slider_ruble = FloatSlider(
    min=cleaned_df['USD_RUB'].min(), max=cleaned_df['USD_RUB'].max(), step=0.1,
    value=cleaned_df['USD_RUB'].mean(), description='USD/RUB (руб):'
)
brent_slider_ruble = FloatSlider(
    min=cleaned_df['Brent'].min(), max=cleaned_df['Brent'].max(), step=0.1,
    value=cleaned_df['Brent'].mean(), description='Brent (долл.):'
)
btc_usd_slider_ruble = FloatSlider(
    min=cleaned_df['BTC_USD'].min(), max=150000, step=100,
    value=cleaned_df['BTC_USD'].mean(), description='BTC/USD (долл.):'
)

# Ползунки для Z-нормализованных данных
btc_mean = cleaned_df['BTC_USD'].mean()
btc_std = cleaned_df['BTC_USD'].std()
btc_max_norm = (150000 - btc_mean) / btc_std
btc_min_norm = (cleaned_df['BTC_USD'].min() - btc_mean) / btc_std

usd_rub_slider_norm = FloatSlider(
    min=normalized_df['USD_RUB'].min(), max=normalized_df['USD_RUB'].max(), step=0.1,
    value=normalized_df['USD_RUB'].mean(), description='USD/RUB (Z-норм):'
)
brent_slider_norm = FloatSlider(
    min=normalized_df['Brent'].min(), max=normalized_df['Brent'].max(), step=0.1,
    value=normalized_df['Brent'].mean(), description='Brent (Z-норм):'
)
btc_usd_slider_norm = FloatSlider(
    min=btc_min_norm, max=btc_max_norm, step=0.1,
    value=normalized_df['BTC_USD'].mean(), description='BTC/USD (Z-норм):'
)

# Переключатели
data_type_dropdown = Dropdown(
    options=['Z-нормализованные', 'Реальные (рубли)'],
    value='Реальные (рубли)',
    description='Тип данных:'
)
model_type_dropdown = Dropdown(
    options=['Линейная регрессия', 'Случайный лес'],
    value='Линейная регрессия',
    description='Модель:'
)

def update_sliders(data_type, model_type):
    if data_type == 'Z-нормализованные':
        interact(
            update_plot,
            usd_rub=usd_rub_slider_norm,
            brent=brent_slider_norm,
            btc_usd=btc_usd_slider_norm,
            data_type=fixed(data_type),
            model_type=fixed(model_type)
        )
    else:
        interact(
            update_plot,
            usd_rub=usd_rub_slider_ruble,
            brent=brent_slider_ruble,
            btc_usd=btc_usd_slider_ruble,
            data_type=fixed(data_type),
            model_type=fixed(model_type)
        )

interact(update_sliders, data_type=data_type_dropdown, model_type=model_type_dropdown)