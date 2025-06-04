import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib

def train_and_save_model():
    # Загрузка данных
    df = pd.read_csv("email_phishing_data.csv")
    
    # Удаление выбросов (оставляем вашу функцию без изменений)
    def locate_outliers(data):
        for col in data.select_dtypes(include=['number']):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            threshold = 1.5
            return data[(data[col] < Q1 - threshold * IQR) | (data[col] > Q3 + threshold * IQR)]
    
    outliers = locate_outliers(df)
    df = df.drop(outliers.index)
    
    # Подготовка данных - используем ТОЧНО те названия, которые вы указали
    features = [
        'num_words', 
        'num_unique_words', 
        'num_stopwords',
        'num_links', 
        'num_unique_domains', 
        'num_email_addresses',
        'num_spelling_errors', 
        'num_urgent_keywords'
    ]
    X = df[features]
    y = df['label']
    
    # Разделение на тренировочную и тестовую выборки (оставляем ваши параметры)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    
    # Масштабирование признаков (без изменений)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обучение модели (оставляем RandomForest без параметров, как у вас)
    model = RandomForestClassifier()
    model.fit(X_train_scaled, y_train)
    
    # Оценка модели (добавляем печать метрик, как у вас было)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Модель обучена с точностью: {accuracy:.2%} и F1-score: {f1:.2%}")
    
    # Сохранение модели, scaler и имен признаков (сохраняем ваш формат)
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': features
    }
    joblib.dump(model_data, 'phishing_model.pkl')
    print("Модель, масштабировщик и имена признаков успешно сохранены")

if __name__ == "__main__":
    train_and_save_model()
