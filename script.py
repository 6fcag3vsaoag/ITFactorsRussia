import pandas as pd
import matplotlib.pyplot as plt

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

# Функция для конвертации строк с числами в float
def convert_to_float(value):
    try:
        # Удаляем точку как разделитель тысяч и заменяем запятую на точку для десятичного разделителя
        cleaned_value = str(value).replace('.', '').replace(',', '.')
        return float(cleaned_value)
    except (ValueError, TypeError):
        return value  # Возвращаем исходное значение, если конвертация не удалась

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