import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_interactive_visualization(normalized_df, results):
    """
    Создает интерактивную визуализацию влияния факторов на акции компаний.
    
    Parameters:
    -----------
    normalized_df : pandas.DataFrame
        DataFrame с нормализованными данными
    results : dict
        Словарь с результатами регрессионного анализа
    """
    def update_plot(usd_rub, brent, btc_rub):
        # Создаем фигуру с подграфиками
        fig = make_subplots(rows=2, cols=2, subplot_titles=('TCSG', 'YDEX', 'VKCO', 'RTKM'))
        
        # Получаем коэффициенты для каждой компании
        companies = ['TCSG', 'YDEX', 'VKCO', 'RTKM']
        positions = [(1,1), (1,2), (2,1), (2,2)]
        
        for company, pos in zip(companies, positions):
            # Получаем коэффициенты из результатов регрессии
            coefs = results[company]['coefficients']
            intercept = results[company]['intercept']
            
            # Рассчитываем предсказанное значение
            predicted = (coefs['USD_RUB'] * usd_rub + 
                        coefs['Brent'] * brent + 
                        coefs['BTC_RUB'] * btc_rub + 
                        intercept)
            
            # Добавляем линию на график
            fig.add_trace(
                go.Scatter(
                    x=[-3, 3],
                    y=[predicted, predicted],
                    mode='lines',
                    name=f'{company} (Предсказано)',
                    line=dict(color='red', width=2)
                ),
                row=pos[0], col=pos[1]
            )
            
            # Добавляем точки реальных данных
            fig.add_trace(
                go.Scatter(
                    x=normalized_df['USD_RUB'],
                    y=normalized_df[company],
                    mode='markers',
                    name=f'{company} (Реальные данные)',
                    marker=dict(color='blue', size=8, opacity=0.5)
                ),
                row=pos[0], col=pos[1]
            )
            
            # Обновляем оси
            fig.update_xaxes(title_text='USD_RUB (Z-score)', row=pos[0], col=pos[1])
            fig.update_yaxes(title_text=f'{company} (Z-score)', row=pos[0], col=pos[1])
        
        fig.update_layout(
            height=800,
            width=1200,
            title_text='Влияние факторов на акции компаний',
            showlegend=True
        )
        
        return fig

    # Создаем ползунки
    usd_slider = widgets.FloatSlider(
        value=0,
        min=-3,
        max=3,
        step=0.1,
        description='USD/RUB:',
        style={'description_width': 'initial'}
    )

    brent_slider = widgets.FloatSlider(
        value=0,
        min=-3,
        max=3,
        step=0.1,
        description='Brent:',
        style={'description_width': 'initial'}
    )

    btc_slider = widgets.FloatSlider(
        value=0,
        min=-3,
        max=3,
        step=0.1,
        description='BTC/RUB:',
        style={'description_width': 'initial'}
    )

    # Создаем интерактивный вывод
    interactive_plot = widgets.interactive_output(
        update_plot,
        {'usd_rub': usd_slider, 'brent': brent_slider, 'btc_rub': btc_slider}
    )

    # Отображаем ползунки и график
    display(widgets.VBox([usd_slider, brent_slider, btc_slider, interactive_plot]))

def create_correlation_heatmap(normalized_df):
    """
    Создает тепловую карту корреляций между всеми факторами.
    
    Parameters:
    -----------
    normalized_df : pandas.DataFrame
        DataFrame с нормализованными данными
    """
    try:
        # Создаем копию DataFrame
        df = normalized_df.copy()
        
        # Удаляем столбец с датами
        if 'Дата' in df.columns:
            df = df.drop('Дата', axis=1)
        
        # Вычисляем корреляции
        corr_matrix = df.corr()
        
        # Создаем тепловую карту
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=corr_matrix.round(2),  # Добавляем текст с значениями корреляций
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Тепловая карта корреляций между факторами',
            height=800,
            width=800,
            xaxis={'side': 'bottom'}  # Размещаем подписи осей снизу
        )
        
        return fig
        
    except Exception as e:
        print(f"Ошибка при создании тепловой карты: {str(e)}")
        print("Доступные столбцы в DataFrame:", df.columns.tolist())
        print("Типы данных столбцов:")
        print(df.dtypes)
        raise

def create_factor_impact_plot(results):
    """
    Создает график влияния каждого фактора на акции компаний.
    
    Parameters:
    -----------
    results : dict
        Словарь с результатами регрессионного анализа
    """
    # Подготавливаем данные для графика
    companies = ['TCSG', 'YDEX', 'VKCO', 'RTKM']
    factors = ['USD_RUB', 'Brent', 'BTC_RUB']
    
    # Создаем DataFrame с коэффициентами
    impact_data = []
    for company in companies:
        coefs = results[company]['coefficients']
        for factor, coef in coefs.items():
            impact_data.append({
                'Компания': company,
                'Фактор': factor,
                'Влияние': coef
            })
    
    impact_df = pd.DataFrame(impact_data)
    
    # Создаем график
    fig = go.Figure()
    
    for company in companies:
        company_data = impact_df[impact_df['Компания'] == company]
        fig.add_trace(go.Bar(
            name=company,
            x=company_data['Фактор'],
            y=company_data['Влияние'],
            text=company_data['Влияние'].round(3),
            textposition='auto',
        ))
    
    fig.update_layout(
        title='Влияние факторов на акции компаний',
        xaxis_title='Факторы',
        yaxis_title='Коэффициент влияния',
        barmode='group',
        height=600,
        width=1000
    )
    
    return fig
