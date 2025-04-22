import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# Понимание данных
def load_data():
    """Загрузить данные из SQLite базы данных"""
    conn = sqlite3.connect('db.sqlite3')
    query = "SELECT * FROM main_app_main_db"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def analyze_data(df):
    """Выполнить начальный анализ данных"""
    print("\nФорма данных:", df.shape)
    print("\nСтолбцы:", df.columns.tolist())
    print("\nТипы данных:\n", df.dtypes)
    print("\nОтсутствующие значения:\n", df.isnull().sum())
    print("\nОсновная статистика:\n", df.describe())

    plt.figure(figsize=(12, 8))
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Матрица корреляции числовых признаков')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()

# Подготовка данных
def prepare_data(df):
    """Подготовка данных для моделирования"""
    # Удаление дубликатов
    df = df.drop_duplicates()
    
    # Выбор признаков
    features = ['estate_type', 'flat_floor', 'building_floor', 'rooms_count',
                'kithcen_square', 'main_square', 'balcony', 'decor',
                'subway_distance', 'apartment_type', 'coordinates_lng', 'coordinates_lat']
    target = 'price'
    
    X = df[features]
    y = df[target]
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Создание предварительных обработок
    numeric_features = ['flat_floor', 'building_floor', 'rooms_count', 'kithcen_square',
                       'main_square', 'subway_distance', 'coordinates_lng', 'coordinates_lat']
    categorical_features = ['estate_type', 'decor', 'apartment_type']
    binary_features = ['balcony']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse=False))
    ])
    
    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('bin', binary_transformer, binary_features)
        ])
    
    return X_train, X_test, y_train, y_test, preprocessor

# Моделирование
def train_model(X_train, y_train, preprocessor):
    """Обучить модель"""
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    return model

# Оценка
def evaluate_model(model, X_test, y_test):
    """Оценить модель"""
    y_pred = model.predict(X_test)
    
    # Вычисление метрик
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\nМетрики оценки модели:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    # Построение графика фактических vs предсказанных значений
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Фактическая цена')
    plt.ylabel('Предсказанная цена')
    plt.title('Фактическая vs Предсказанная цена')
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    plt.close()

def main():
    # Загрузка данных
    print("Загрузка данных...")
    df = load_data()
    
    # Анализ данных
    print("Анализ данных...")
    analyze_data(df)
    
    # Подготовка данных
    print("Подготовка данных...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)
    
    # Моделирование
    print("Моделирование...")
    model = train_model(X_train, y_train, preprocessor)
    
    # Оценка
    print("Оценка...")
    evaluate_model(model, X_test, y_test)
    
    print("\nАнализ завершен! Проверьте correlation_matrix.png и actual_vs_predicted.png для визуализаций.")

if __name__ == "__main__":
    main() 