import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from PIL import Image, ImageTk
import io

class StockAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BRU Money Hustler")
        self.root.geometry("1230x700")
        self.root.minsize(900, 700)
        
        # Глобальные переменные
        self.final_df = None
        self.cleaned_df = None
        self.normalized_df = None
        self.results_normalized = {}
        self.results_normalized_rf = {}
        self.analysis_complete = False
        self.current_plot = None
        
        # Инициализация интерфейса
        self.setup_ui()
        
        # Запуск анализа в фоновом потоке
        self.start_analysis()
    
    def setup_ui(self):
        """Настройка пользовательского интерфейса с улучшенным дизайном"""
        # Конфигурация стилей
        style = ttk.Style()
        style.configure('TFrame', background='#f5f5f5')
        style.configure('TLabel', background='#f5f5f5', font=('Helvetica', 10))
        style.configure('TButton', font=('Helvetica', 10))
        style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))
        style.configure('TScale', background='#f5f5f5')
        
        # Главный контейнер
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Левая панель - управление
        self.control_panel = ttk.Frame(self.main_container, width=350, relief=tk.RAISED, borderwidth=1)
        self.control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        self.control_panel.pack_propagate(False)
        
        # Правая панель - график
        self.plot_panel = ttk.Frame(self.main_container)
        self.plot_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Заголовок панели управления
        ttk.Label(self.control_panel, text="Параметры прогноза", style='Header.TLabel').pack(pady=(10, 20))
        
        # Индикатор загрузки
        self.loading_frame = ttk.Frame(self.control_panel)
        self.loading_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.loading_label = ttk.Label(self.loading_frame, text="Выполняется анализ данных...")
        self.loading_label.pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(self.loading_frame, mode='indeterminate')
        self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=10)
        self.progress.start()
        
        # Переменные для элементов управления
        self.data_type_var = tk.StringVar(value='Реальные (RUB)')
        self.model_type_var = tk.StringVar(value='Линейная регрессия')
        self.usd_rub_var = tk.DoubleVar(value=80)
        self.brent_var = tk.DoubleVar(value=70)
        self.btc_usd_var = tk.DoubleVar(value=30000)
        
        # Секция выбора модели
        self.create_model_section()
        
        # Секция слайдеров
        self.create_sliders_section()
        
        # Секция сообщения о текущем прогнозе
        self.create_forecast_message_section()
        
        # Секция для логотипа
        self.create_logo_section()
        
        # Отображение графика
        self.plot_label = ttk.Label(self.plot_panel)
        self.plot_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Привязка событий
        self.data_type_var.trace_add('write', self.on_data_type_change)
        self.model_type_var.trace_add('write', lambda *args: self.update_plot())
        self.usd_rub_var.trace_add('write', self.on_slider_change)
        self.brent_var.trace_add('write', self.on_slider_change)
        self.btc_usd_var.trace_add('write', self.on_slider_change)
    
    def create_model_section(self):
        """Создание секции выбора модели"""
        model_frame = ttk.LabelFrame(self.control_panel, text="Настройки модели", padding=(15, 10))
        model_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Выбор типа данных
        ttk.Label(model_frame, text="Тип данных:").grid(row=0, column=0, sticky='w', pady=(0, 10))
        self.data_type_menu = ttk.Combobox(model_frame, textvariable=self.data_type_var, 
                                         values=['Z-нормализованные', 'Реальные (RUB)'], state='disabled')
        self.data_type_menu.grid(row=0, column=1, sticky='ew', pady=(0, 10), padx=(10, 0))
        
        # Выбор модели
        ttk.Label(model_frame, text="Модель прогноза:").grid(row=1, column=0, sticky='w', pady=(0, 10))
        self.model_type_menu = ttk.Combobox(model_frame, textvariable=self.model_type_var, 
                                          values=['Линейная регрессия', 'Случайный лес'], state='disabled')
        self.model_type_menu.grid(row=1, column=1, sticky='ew', pady=(0, 10), padx=(10, 0))
        
        # Конфигурация сетки
        model_frame.columnconfigure(1, weight=1)
    
    def create_sliders_section(self):
        """Создание секции слайдеров с улучшенным выравниванием и дизайном"""
        sliders_frame = ttk.LabelFrame(self.control_panel, text="Факторы влияния", padding=(15, 15))
        sliders_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Создание сетки для согласованного выравнивания
        sliders = [
            ("USD/RUB (RUB):", self.usd_rub_var, 'usd_rub'),
            ("Brent (USD):", self.brent_var, 'brent'),
            ("BTC/USD (USD):", self.btc_usd_var, 'btc_usd')
        ]
        
        for idx, (label_text, var, prefix) in enumerate(sliders):
            # Контейнер для каждой группы слайдеров
            slider_container = ttk.Frame(sliders_frame)
            slider_container.pack(fill=tk.X, pady=(10 if idx > 0 else 0, 10))
            
            # Метка
            label = ttk.Label(slider_container, text=label_text, width=15)
            label.pack(side=tk.LEFT, padx=(0, 10))
            
            # Отображение значения
            value_label = ttk.Label(slider_container, text="0.00", width=10, anchor='e')
            value_label.pack(side=tk.RIGHT, padx=(10, 0))
            
            # Слайдер
            scale = ttk.Scale(slider_container, variable=var, orient=tk.HORIZONTAL, state='disabled')
            scale.pack(fill=tk.X, expand=True)
            
            # Сохранение ссылок
            setattr(self, f"{prefix}_scale", scale)
            setattr(self, f"{prefix}_value_label", value_label)
    
    def create_forecast_message_section(self):
        """Создание секции для отображения текущего прогноза"""
        forecast_frame = ttk.LabelFrame(self.control_panel, text="Текущие цены по активам", padding=(15, 10))
        forecast_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.forecast_label = ttk.Label(forecast_frame, text="Обновите параметры для просмотра цен.", 
                                      justify='left', wraplength=300, font=('Helvetica', 10))
        self.forecast_label.pack(padx=5, pady=5)
    
    def create_logo_section(self):
        """Создание секции для отображения логотипа"""
        logo_frame = ttk.Frame(self.control_panel)
        logo_frame.pack(fill=tk.X, pady=(0, 10))
        
        try:
            # Загрузка и изменение размера логотипа
            logo_img = Image.open("logo.webp")
            logo_img = logo_img.resize((350, 100), Image.LANCZOS)  # Размер логотипа подогнан под панель
            self.logo_tk = ImageTk.PhotoImage(logo_img)
            
            # Отображение логотипа
            logo_label = ttk.Label(logo_frame, image=self.logo_tk, background='#f5f5f5')
            logo_label.pack(pady=5)
        except Exception as e:
            print(f"Ошибка при загрузке логотипа: {e}")
            # Запасной текст, если логотип не загрузился
            ttk.Label(logo_frame, text="Логотип не загружен", font=('Helvetica', 10)).pack(pady=5)
    
    def on_data_type_change(self, *args):
        """Обработчик изменения типа данных"""
        if self.analysis_complete:
            self.update_sliders()
            self.update_plot()
    
    def on_slider_change(self, *args):
        """Обработчик изменения слайдеров"""
        self.update_slider_labels()
        if self.analysis_complete:
            self.update_plot()
    
    def enable_controls(self):
        """Активация элементов управления после завершения анализа"""
        self.loading_frame.pack_forget()
        
        # Активируем элементы управления
        self.data_type_menu.config(state='readonly')
        self.model_type_menu.config(state='readonly')
        self.usd_rub_scale.config(state='normal')
        self.brent_scale.config(state='normal')
        self.btc_usd_scale.config(state='normal')
        
        # Обновляем слайдеры
        self.update_sliders()
        self.update_slider_labels()
        
        # Первое обновление графика
        self.update_plot()
    
    def start_analysis(self):
        """Запуск анализа в фоновом потоке"""
        def analysis_thread():
            try:
                self.perform_analysis()
                self.analysis_complete = True
                self.root.after(0, self.enable_controls)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Ошибка", f"Ошибка при анализе данных: {str(e)}"))
        
        threading.Thread(target=analysis_thread, daemon=True).start()
    
    def convert_to_float(self, value):
        """Конвертация строк с числами в float"""
        try:
            cleaned_value = str(value).replace('.', '').replace(',', '.')
            return float(cleaned_value)
        except (ValueError, TypeError):
            return value
    
    def remove_outliers(self, df, columns):
        """Удаление выбросов по методу IQR"""
        mask = pd.Series(True, index=df.index)
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            mask = mask & (df[col].between(lower_bound, upper_bound))
        return df[mask]
    
    def perform_analysis(self):
        """Основная функция анализа данных"""
        # 1. Чтение и объединение данных
        files = {
            'tksg.csv': 'TCSG',
            'yandex.csv': 'YDEX',
            'vk.csv': 'VKCO',
            'rostelecom.csv': 'RTKM',
            'usd_rub.csv': 'USD_RUB',
            'brent.csv': 'Brent',
            'bitcoin.csv': 'BTC_USD'
        }
        
        dfs = []
        for file_name, column_name in files.items():
            try:
                df = pd.read_csv(file_name)
                df = df[['Дата', 'Цена']].rename(columns={'Цена': column_name})
                dfs.append(df)
            except Exception as e:
                print(f"Ошибка при чтении файла {file_name}: {str(e)}")
        
        if not dfs:
            raise ValueError("Не удалось загрузить ни один файл данных")
        
        # Объединение данных
        self.final_df = dfs[0]
        for df in dfs[1:]:
            self.final_df = pd.merge(self.final_df, df, on='Дата', how='inner')
        
        # Обработка данных
        self.final_df['Дата'] = pd.to_datetime(self.final_df['Дата'], format='%d.%m.%Y')
        self.final_df = self.final_df.sort_values('Дата')
        
        for column in self.final_df.columns:
            if column != 'Дата':
                self.final_df[column] = self.final_df[column].apply(self.convert_to_float)
        
        # 2. Очистка от выбросов
        columns_to_clean = list(files.values())
        self.cleaned_df = self.remove_outliers(self.final_df, columns_to_clean)
        
        # 3. Z-нормализация
        self.normalized_df = self.cleaned_df.copy()
        for col in columns_to_clean:
            mean = self.cleaned_df[col].mean()
            std = self.cleaned_df[col].std()
            self.normalized_df[col] = (self.cleaned_df[col] - mean) / std
        
        # 5. Анализ регрессии
        dependent_vars = ['TCSG', 'YDEX', 'VKCO', 'RTKM']
        independent_vars = ['USD_RUB', 'Brent', 'BTC_USD']
        X_normalized = self.normalized_df[independent_vars].dropna()
        
        for y_var in dependent_vars:
            y = self.normalized_df[y_var].dropna()
            common_index = X_normalized.index.intersection(y.index)
            X_subset = X_normalized.loc[common_index]
            y_subset = y.loc[common_index]
            
            if len(X_subset) == 0 or len(y_subset) == 0:
                print(f"\nНедостаточно данных для {y_var}")
                continue
            
            # Исключение фактора Brent для TCSG
            if y_var == 'TCSG':
                X_subset = X_subset[['USD_RUB', 'BTC_USD']]
            
            X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)
            
            # Линейная регрессия
            model_lr = LinearRegression()
            model_lr.fit(X_train, y_train)
            y_pred_lr = model_lr.predict(X_test)
            mae_lr = np.mean(np.abs(y_test - y_pred_lr))
            mse_lr = np.mean((y_test - y_pred_lr) ** 2)
            
            self.results_normalized[y_var] = {
                'y_actual': y_test,
                'y_pred': y_pred_lr,
                'X_test': X_test,
                'model': model_lr,
                'coefficients': dict(zip(X_subset.columns, model_lr.coef_)),
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
            
            self.results_normalized_rf[y_var] = {
                'y_actual': y_test,
                'y_pred': y_pred_rf,
                'X_test': X_test,
                'model': model_rf,
                'feature_importances': dict(zip(X_subset.columns, model_rf.feature_importances_)),
                'MAE': mae_rf,
                'MSE': mse_rf
            }
    
    def update_sliders(self):
        """Обновление диапазонов слайдеров"""
        if not self.analysis_complete:
            return
            
        if self.data_type_var.get() == 'Z-нормализованные':
            # Для нормализованных данных
            self.usd_rub_scale.configure(
                from_=self.normalized_df['USD_RUB'].min(),
                to=self.normalized_df['USD_RUB'].max()
            )
            self.usd_rub_var.set(self.normalized_df['USD_RUB'].mean())
            
            self.brent_scale.configure(
                from_=self.normalized_df['Brent'].min(),
                to=self.normalized_df['Brent'].max()
            )
            self.brent_var.set(self.normalized_df['Brent'].mean())
            
            btc_mean = self.cleaned_df['BTC_USD'].mean()
            btc_std = self.cleaned_df['BTC_USD'].std()
            btc_max_norm = (150000 - btc_mean) / btc_std
            btc_min_norm = (self.cleaned_df['BTC_USD'].min() - btc_mean) / btc_std
            
            self.btc_usd_scale.configure(
                from_=btc_min_norm,
                to=btc_max_norm
            )
            self.btc_usd_var.set(self.normalized_df['BTC_USD'].mean())
        else:
            # Для реальных значений
            self.usd_rub_scale.configure(
                from_=self.cleaned_df['USD_RUB'].min(),
                to=self.cleaned_df['USD_RUB'].max()
            )
            self.usd_rub_var.set(self.cleaned_df['USD_RUB'].mean())
            
            self.brent_scale.configure(
                from_=self.cleaned_df['Brent'].min(),
                to=self.cleaned_df['Brent'].max()
            )
            self.brent_var.set(self.cleaned_df['Brent'].mean())
            
            self.btc_usd_scale.configure(
                from_=self.cleaned_df['BTC_USD'].min(),
                to=150000
            )
            self.btc_usd_var.set(self.cleaned_df['BTC_USD'].mean())
    
    def update_slider_labels(self):
        """Обновление меток значений слайдеров"""
        self.usd_rub_value_label.config(text=f"{self.usd_rub_var.get():.2f}")
        self.brent_value_label.config(text=f"{self.brent_var.get():.2f}")
        self.btc_usd_value_label.config(text=f"{self.btc_usd_var.get():.2f}")
    
    def forecast_prices(self, usd_rub, brent, btc_usd, data_type, model_type):
        """Прогнозирование цен на основе введенных данных"""
        if not self.analysis_complete:
            return {}, 'Цена акций'
            
        try:
            prices = {}
            results = self.results_normalized if model_type == 'Линейная регрессия' else self.results_normalized_rf
            ylabel = 'Цена акций (Z-нормализованная)' if data_type == 'Z-нормализованные' else 'Цена акций (RUB)'
            
            # Нормализация входных данных, если нужно
            if data_type == 'Реальные (RUB)':
                usd_rub_norm = (usd_rub - self.cleaned_df['USD_RUB'].mean()) / self.cleaned_df['USD_RUB'].std()
                brent_norm = (brent - self.cleaned_df['Brent'].mean()) / self.cleaned_df['Brent'].std()
                btc_usd_norm = (btc_usd - self.cleaned_df['BTC_USD'].mean()) / self.cleaned_df['BTC_USD'].std()
            else:
                usd_rub_norm = usd_rub
                brent_norm = brent
                btc_usd_norm = btc_usd
            
            for y_var in ['TCSG', 'YDEX', 'VKCO', 'RTKM']:
                if y_var not in results:
                    continue
                    
                model = results[y_var]['model']
                expected_features = list(results[y_var]['X_test'].columns)
                
                # Подготовка входных данных
                input_data = {}
                if 'USD_RUB' in expected_features:
                    input_data['USD_RUB'] = [usd_rub_norm]
                if 'Brent' in expected_features and y_var != 'TCSG':
                    input_data['Brent'] = [brent_norm]
                if 'BTC_USD' in expected_features:
                    input_data['BTC_USD'] = [btc_usd_norm]
                
                input_df = pd.DataFrame(input_data)
                price = model.predict(input_df)[0]
                
                # Денормализация, если нужно
                if data_type == 'Реальные (RUB)':
                    price = price * self.cleaned_df[y_var].std() + self.cleaned_df[y_var].mean()
                    price = max(0, price)
                
                prices[y_var] = price
            
            return prices, ylabel
        except Exception as e:
            print(f"Ошибка при прогнозировании: {e}")
            return {}, 'Цена акций'
    
    def update_plot(self):
        """Обновление графика прогнозов"""
        if not self.analysis_complete:
            return
            
        data_type = self.data_type_var.get()
        model_type = self.model_type_var.get()
        usd_rub = self.usd_rub_var.get()
        brent = self.brent_var.get()
        btc_usd = self.btc_usd_var.get()
        
        prices, ylabel = self.forecast_prices(usd_rub, brent, btc_usd, data_type, model_type)
        
        if not prices:
            return
        
        # Создание графика в памяти
        plt.figure(figsize=(10, 8))
        companies = list(prices.keys())
        values = [prices[company] for company in companies]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = plt.bar(companies, values, color=colors)
        
        plt.title(f'Прогнозируемые цены акций ({data_type}, {model_type})', fontsize=14)
        plt.xlabel('Компания')
        plt.ylabel(ylabel)
        plt.grid(True, axis='y')
        
        # Добавление значений на столбцы
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        self.current_plot = plt.gcf()
        plt.close()
        
        # Обновление отображения
        try:
            buf = io.BytesIO()
            self.current_plot.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            img = img.resize((800, 600), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            self.plot_label.configure(image=img_tk)
            self.plot_label.image = img_tk
            buf.close()
            
            # Обновление сообщения о прогнозе
            self.update_forecast_message(prices)
        except Exception as e:
            print(f"Ошибка при отображении графика: {e}")

    def update_forecast_message(self, prices):
        """Обновление сообщения о текущем прогнозе"""
        if not prices:
            self.forecast_label.config(text="Обновите параметры для просмотра цен.")
            return
        
        message = "Текущие прогнозируемые цены:\n\n"
        for company, price in prices.items():
            message += f"{company}: {price:.2f} RUB\n"
        self.forecast_label.config(text=message, justify='left')

if __name__ == "__main__":
    root = tk.Tk()
    app = StockAnalysisApp(root)
    root.mainloop()