import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import streamlit as st
import time
from functools import lru_cache

# Кэшируем обучение модели для быстрой загрузки
@st.cache_data
def train_model():
    # Загрузка данных
    df = pd.read_csv("email_phishing_data.csv")
    
    # Удаление выбросов
    def locate_outliers(data):
        for col in data.select_dtypes(include=['number']):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            threshold = 1.5
            return data[(data[col] < Q1 - threshold * IQR) | (data[col] > Q3 + threshold * IQR)]
    
    outliers = locate_outliers(df)
    df = df.drop(outliers.index)
    
    # Подготовка данных
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
    
    # Разделение на тренировочную и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    
    # Масштабирование признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обучение модели
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Оценка модели
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return model, scaler, features, accuracy, f1

# Функция для расчета риска в реальном времени
def calculate_risk(model, scaler, features, input_values):
    input_df = pd.DataFrame([input_values], columns=features)
    scaled_data = scaler.transform(input_df)
    return model.predict_proba(scaled_data)[0][1]

# Основная функция приложения
def main():
    # Настройки страницы
    st.set_page_config(
        page_title="PhishShield PRO",
        page_icon="🛡️",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # Стилизация
    st.markdown("""
    <style>
        .header-style {
            font-size: 24px;
            font-weight: bold;
            color: #2e86c1;
        }
        .metric-value {
            font-size: 28px !important;
            font-weight: bold !important;
        }
        .stSpinner > div {
            text-align: center;
            margin-top: 10px;
        }
        .risk-indicator {
            transition: all 0.5s ease;
        }
        .pulse-animation {
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
    </style>
    """, unsafe_allow_html=True)

    # Загрузка и обучение модели с индикатором прогресса
    with st.spinner('Загрузка модели... Это займет всего несколько секунд'):
        model, scaler, features, accuracy, f1 = train_model()

    # Интерфейс приложения
    st.markdown('<p class="header-style">PhishShield PRO</p>', unsafe_allow_html=True)
    st.caption("Детектор фишинговых писем с порогом обнаружения 30%")

    # Форма ввода параметров
    with st.form("phish_form"):
        st.subheader("Параметры письма")
        
        input_values = {}
        cols = st.columns(2)
        
        with cols[0]:
            input_values['num_words'] = st.number_input("Общее количество слов", min_value=0, value=120, key='num_words')
            input_values['num_unique_words'] = st.number_input("Уникальные слова", min_value=0, value=80, key='num_unique_words')
            input_values['num_stopwords'] = st.number_input("Стоп-слова", min_value=0, value=30, key='num_stopwords')
            input_values['num_links'] = st.number_input("Ссылки", min_value=0, value=2, key='num_links')
        
        with cols[1]:
            input_values['num_unique_domains'] = st.number_input("Уникальные домены", min_value=0, value=1, key='num_unique_domains')
            input_values['num_email_addresses'] = st.number_input("Email адреса", min_value=0, value=1, key='num_email_addresses')
            input_values['num_spelling_errors'] = st.number_input("Ошибки", min_value=0, value=3, key='num_spelling_errors')
            input_values['num_urgent_keywords'] = st.number_input("Срочность", min_value=0, value=5, key='num_urgent_keywords')
        
        submitted = st.form_submit_button("Анализировать", type="primary")

    # Контейнер для динамического отображения риска
    risk_display = st.empty()
    
    # Расчет риска в реальном времени
    if any(input_values.values()):  # Если есть введенные значения
        try:
            phishing_prob = calculate_risk(model, scaler, features, input_values)
            threshold = 0.3
            risk_level = "Высокий" if phishing_prob >= threshold else "Низкий"
            risk_color = "#e74c3c" if phishing_prob >= threshold else "#2ecc71"
            
            # Анимация пульсации для высокого риска
            pulse_class = "pulse-animation" if phishing_prob >= threshold else ""
            
            # Динамическое отображение риска
            with risk_display.container():
                st.subheader("Текущий риск фишинга")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"""
                    <div class="risk-indicator {pulse_class}" style="margin-bottom: 1rem;">
                        <div style="font-size: 16px; color: #7f8c8d;">Вероятность фишинга</div>
                        <div class="metric-value" style="color: {risk_color};">{phishing_prob*100:.1f}%</div>
                        <div style="font-size: 14px; color: #7f8c8d;">Порог: {threshold*100:.0f}%</div>
                        <div style="font-size: 14px; color: {risk_color}; font-weight: bold;">{risk_level} риск</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Прогресс-бар с анимацией
                    progress_bar = st.progress(0)
                    for percent_complete in range(int(phishing_prob*100)):
                        time.sleep(0.01)
                        progress_bar.progress(percent_complete + 1)
                    progress_bar.progress(float(phishing_prob))

        except Exception as e:
            st.error(f"Ошибка анализа: {str(e)}")

    # Обработка результатов при нажатии кнопки
    if submitted:
        st.success("Анализ завершен! Результаты обновлены выше.")
        st.balloons()  # Эффект праздничных шариков для визуального подтверждения

    # Боковая панель с дополнительной информацией
    with st.sidebar:
        st.header("ℹ️ О системе")
        st.info("""
        **PhishShield PRO** использует машинное обучение
        для анализа писем на признаки фишинга.
        Порог срабатывания: 30% вероятности.
        """)
        
        st.markdown("---")
        st.write("📊 **Метрики модели:**")
        st.metric("Точность", f"{accuracy:.2%}")
        st.metric("F1-score", f"{f1:.2%}")
        
        st.markdown("---")
        st.write("**📌 Как использовать:**")
        st.write("1. Введите параметры письма")
        st.write("2. Наблюдайте за изменением риска в реальном времени")
        st.write("3. Нажмите 'Анализировать' для фиксации результата")
        
        st.markdown("---")
        st.write("**🔍 Типичные признаки фишинга:**")
        st.write("- Срочные просьбы или угрозы")
        st.write("- Подозрительные ссылки/вложения")
        st.write("- Ошибки в тексте письма")
        st.write("- Незнакомые отправители")
        st.write("- Запрос личной информации")

if __name__ == "__main__":
    main()