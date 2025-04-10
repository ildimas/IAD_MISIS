# Импорт библиотек
import pandas as pd  # Для работы с табличными данными
import dash  # Фреймворк для создания веб-приложений
from dash import dcc, html  # Компоненты Dash: элементы интерфейса и графика
import plotly.express as px  # Библиотека для построения интерактивных графиков

# Задание пути к Excel-файлу с ESG-отчетностью
excel_path = "PhosAgro_ESG_databook-Rus.xlsx"

# Загрузка всех листов Excel-файла в словарь (ключ — имя листа, значение — DataFrame)
df = pd.read_excel(excel_path, sheet_name=None, engine="openpyxl")

# Извлечение листа "КОРПОРАТИВНОЕ УПРАВЛЕНИЕ"
corporate_governance_df = df.get("КОРПОРАТИВНОЕ УПРАВЛЕНИЕ")

# Список интересующих метрик (строк таблицы)
metrics_of_interest = [
    "Доля независимых директоров в составе Комитета",
    "Количество заседаний комитета",
    "Количество членов Правления",
    "Количество заседаний Правления",
    "Общее количество обращений, поступивших на портал «Горячей линии «ФосАгро»",
    "Количество обращений, связанных с проявлением коррупции ",
    "Размер вознаграждения аудитора ",
    "Доля сотрудников, ознакомленных с требованиями/политиками ФосАгро по противодействию коррупции",
]

# Фильтрация строк таблицы: оставляем только строки с метриками из списка
filtered_corporate_governance_df = corporate_governance_df[
    corporate_governance_df.iloc[:, 0].isin(metrics_of_interest)
]

# Инициализация Dash-приложения
app = dash.Dash(__name__)
app.title = "Corporate Governance Dashboard"  # Название вкладки браузера

# Определение интерфейса приложения
app.layout = html.Div([
    html.H1("📊 Корпоративный дэшборд", style={'textAlign': 'center'}),  # Заголовок по центру
    dcc.Dropdown(
        id="metric-selector",  # ID выпадающего списка
        options=[{"label": metric, "value": metric} for metric in metrics_of_interest],  # Пункты выбора
        value=metrics_of_interest[0],  # Значение по умолчанию
        style={"width": "80%", "margin": "0 auto"}  # Стилизация по ширине и выравниванию
    ),
    dcc.Graph(id="line-chart")  # График для отображения данных
])

# Обратный вызов (callback) — обновление графика при выборе другой метрики
@app.callback(
    dash.dependencies.Output("line-chart", "figure"),  # Обновляемый элемент — график
    [dash.dependencies.Input("metric-selector", "value")]  # Входной элемент — выбор метрики
)
def update_chart(selected_metric):
    # Поиск строки по выбранной метрике
    row = filtered_corporate_governance_df[
        filtered_corporate_governance_df.iloc[:, 0] == selected_metric
    ].iloc[0]

    # Задаем годы и значения (предположительно, они находятся в колонках с 3 по 7)
    years = ["2019", "2020", "2021", "2022", "2023"]
    values = row.iloc[2:7].tolist()

    # Построение линейного графика
    fig = px.line(x=years, y=values, markers=True, title=f"{selected_metric} (2019–2023)")
    fig.update_layout(xaxis_title="Year", yaxis_title="Value", template="plotly_white")
    return fig  # Возвращаем обновленный график

# Запуск приложения в режиме отладки (debug=True)
if __name__ == "__main__":
    app.run(debug=True)
