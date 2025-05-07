import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Настройка логирования
logging.basicConfig(filename="log.txt", level=logging.INFO)

def log_action(action):
    logging.info(f"{action} | {pd.Timestamp.now()}")

st.set_page_config(page_title="🧼 Чистка и Анализ Данных", layout="wide")
st.title("📊 Чистка и анализ данных")

# Функции
def try_read_csv(file):
    """Попытка прочитать CSV с несколькими кодировками"""
    encodings = ['utf-8', 'utf-16', 'cp1251', 'ISO-8859-1']
    for enc in encodings:
        try:
            return pd.read_csv(file, encoding=enc)
        except Exception:
            continue
    raise ValueError("❌ Не удалось прочитать CSV с поддерживаемыми кодировками")

def save_df_to_file(dataframe, filename):
    filepath = os.path.join("data", filename)
    dataframe.to_csv(filepath, index=False)
    return filepath

# Загрузка данных
st.header("📁 Загрузка данных")
df = None
upload_method = st.radio("Выберите способ загрузки", ["Из папки data/", "Загрузить файл"])
if upload_method == "Из папки data/":
    files = [f for f in os.listdir("data") if f.endswith((".csv", ".xlsx", ".xls"))]
    if files:
        file_choice = st.selectbox("Выберите файл", files)
        path = os.path.join("data", file_choice)
        try:
            if file_choice.endswith(".csv"):
                df = try_read_csv(path)
            else:
                df = pd.read_excel(path)
            log_action(f"Загружен файл: {file_choice}")
        except Exception as e:
            st.error(f"❌ Ошибка при чтении файла: {e}")
    else:
        st.warning("⚠️ В папке data/ нет файлов.")
else:
    uploaded_file = st.file_uploader("Загрузите CSV или Excel", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = try_read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            log_action(f"Загружен файл: {uploaded_file.name}")
        except Exception as e:
            st.error(f"❌ Ошибка при чтении файла: {e}")

if df is not None:
    st.subheader("📝 Просмотр данных")
    st.dataframe(df)

    # Фильтрация
    if st.checkbox("🔍 Фильтровать данные"):
        col = st.selectbox("Колонка", df.columns)
        vals = st.multiselect("Значения", df[col].unique())
        if vals:
            df = df[df[col].isin(vals)]
            log_action(f"Фильтрация по {col}: {vals}")

    # Пропуски
    st.subheader("❓ Пропущенные значения")
    if df.isnull().sum().sum() > 0:
        st.write(df.isnull().sum())
        method = st.radio("Заполнить пропуски методом:", ["Среднее", "Медиана", "Удалить строки", "Ничего не делать"])
        if method == "Среднее":
            df = df.fillna(df.mean(numeric_only=True))
            st.success("✅ Заполнено средними")
            log_action("Заполнение пропусков: среднее")
        elif method == "Медиана":
            df = df.fillna(df.median(numeric_only=True))
            st.success("✅ Заполнено медианой")
            log_action("Заполнение пропусков: медиана")
        elif method == "Удалить строки":
            df = df.dropna()
            st.success("✅ Удалены строки с пропусками")
            log_action("Удалены строки с пропущенными значениями")
    else:
        st.success("✅ Нет пропущенных значений")

    # Удаление дубликатов
    if st.checkbox("🧹 Удалить дубликаты"):
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        st.success(f"✅ Удалено {before - after} дубликатов")
        log_action(f"Удалено {before - after} дубликатов")

    # Выбросы
    if st.checkbox("📦 Посмотреть выбросы (Boxplot)"):
        col = st.selectbox("Выберите числовую колонку", df.select_dtypes(include=np.number).columns)
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        st.pyplot(fig)

    # Агрегация и визуализация
    st.subheader("📈 Группировка и визуализация")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    group_col = st.selectbox("Группировать по", df.columns)
    agg_col = st.selectbox("Агрегировать колонку", numeric_cols)
    agg_func = st.selectbox("Функция", ["mean", "sum", "count", "min", "max"])
    grouped = df.groupby(group_col)[agg_col].agg(agg_func).reset_index()
    st.write(grouped)

    # Визуализация
    chart_type = st.radio("Тип графика", ["Bar", "Line", "Pie"])
    fig, ax = plt.subplots()
    if chart_type == "Bar":
        ax.bar(grouped[group_col], grouped[agg_col])
    elif chart_type == "Line":
        ax.plot(grouped[group_col], grouped[agg_col])
    elif chart_type == "Pie":
        ax.pie(grouped[agg_col], labels=grouped[group_col], autopct="%1.1f%%")
    st.pyplot(fig)

    # Сохранение
    st.subheader("💾 Сохранить результат")
    file_name = st.text_input("Имя файла", "result.csv")
    if st.button("📥 Скачать CSV"):
        st.download_button("⬇️ Скачать", df.to_csv(index=False), file_name, "text/csv")
        log_action(f"Скачан CSV: {file_name}")
    if st.button("💾 Сохранить в папку data/"):
        path = save_df_to_file(df, file_name)
        st.success(f"✅ Сохранено: {path}")
        log_action(f"Сохранено в папку: {file_name}")