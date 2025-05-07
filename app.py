import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from io import StringIO
from scipy.stats import zscore

# === 1. Загрузка данных ===
def load_data():
    st.title("🧹 Очистка и анализ данных")
    method = st.radio("Выберите способ загрузки", [
        "Из списка файлов в папке",
        "Загрузить файл с компьютера"
    ])
    df = None

    if method == "Из списка файлов в папке":
        files = [f for f in os.listdir() if f.endswith(".csv")]
        if not files:
            st.warning("❌ В папке нет CSV-файлов.")
            return None
        file_selected = st.selectbox("Выберите файл", files)
        try:
            df = pd.read_csv(file_selected, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file_selected, encoding="ISO-8859-1")
        except Exception as e:
            st.error(f"❌ Ошибка при загрузке: {e}")
            return None

    elif method == "Загрузить файл с компьютера":
        uploaded_file = st.file_uploader("Загрузите CSV-файл", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
            except Exception as e:
                st.error(f"❌ Ошибка при чтении файла: {e}")
                return None

    if df is not None:
        if df.empty:
            st.warning("⚠️ Загруженный файл пустой.")
            return None

        df.columns = df.columns.str.strip()
        st.success("✅ Данные успешно загружены")
        st.write("📊 Первые 5 строк данных")
        st.dataframe(df.head())

        st.write("ℹ️ Информация о DataFrame")
        buffer = StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        return df

    return None

# === 2. Просмотр данных ===
def view_data(df):
    st.subheader("🔍 Просмотр данных")
    st.write("Размер данных:", df.shape)
    st.dataframe(df.head())

# === 3. Анализ пропусков ===
def show_missing(df):
    st.subheader("📉 Анализ пропущенных значений")
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing == 0:
        st.success("✅ Нет пропущенных значений")
    else:
        st.warning("⚠️ Пропущенные значения:")
        st.dataframe(missing[missing > 0])
    return total_missing

# === 4. Заполнение пропусков вручную ===
def fill_missing(df):
    st.subheader("🧩 Заполнение пропусков вручную")
    for col in df.columns[df.isnull().any()]:
        st.write(f"📌 Обработка колонки: **{col}**")
        if df[col].dtype == "object":
            fill_value = st.text_input(f"Введите значение для заполнения (строка):", key=col)
            if fill_value:
                df[col].fillna(fill_value, inplace=True)
        else:
            method = st.selectbox(f"Метод заполнения для {col}", ["среднее", "медиана", "мода"], key=col)
            if method == "среднее":
                df[col].fillna(df[col].mean(), inplace=True)
            elif method == "медиана":
                df[col].fillna(df[col].median(), inplace=True)
            elif method == "мода":
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col].fillna(mode_val[0], inplace=True)
    return df

# === 5. Автоматическое заполнение пропусков ===
def auto_fill_missing(df):
    st.subheader("⚙️ Автоматическая обработка пропусков")
    for col in df.columns[df.isnull().any()]:
        if df[col].dtype == "object":
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col].fillna(mode_val[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
    st.success("✅ Пропуски автоматически обработаны")
    return df

# === 6. Удаление дубликатов ===
def remove_duplicates(df):
    st.subheader("🗑 Удаление дубликатов")
    original = df.shape[0]
    df = df.drop_duplicates()
    removed = original - df.shape[0]
    st.success(f"✅ Удалено дубликатов: {removed} строк")
    return df

# === 7. Удаление выбросов ===
def remove_outliers(df):
    st.subheader("📉 Удаление выбросов")
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) == 0:
        st.info("⚠️ В таблице нет числовых колонок для анализа выбросов.")
        return df

    col = st.selectbox("Выберите числовую колонку для удаления выбросов", numeric_cols)
    method = st.selectbox("Метод определения выбросов", ["IQR (межквартильный размах)", "Z-score"])

    original_size = df.shape[0]

    if method == "IQR (межквартильный размах)":
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    elif method == "Z-score":
        z_scores = np.abs(zscore(df[col]))
        df = df[z_scores < 3]

    removed = original_size - df.shape[0]
    st.success(f"✅ Удалено выбросов: {removed} строк")
    return df

# === 8. Визуализация данных ===
def visualize(df):
    st.subheader("📊 Визуализация")
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) < 2:
        st.warning("⚠️ Недостаточно числовых колонок для визуализации.")
        return

    x_col = st.selectbox("Ось X", numeric_cols, key="x")
    y_col = st.selectbox("Ось Y", numeric_cols, key="y")

    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=df, x=x_col, y=y_col)
    st.pyplot(plt)

# === 9. Главная функция ===
def main():
    df = load_data()
    if df is not None:
        view_data(df)

        if show_missing(df) > 0:
            if st.checkbox("🔧 Обработать пропущенные значения вручную"):
                df = fill_missing(df)

            if st.checkbox("🤖 Автоматически обработать все пропуски"):
                df = auto_fill_missing(df)

        if st.checkbox("🗑 Удалить дубликаты"):
            df = remove_duplicates(df)

        if st.checkbox("📉 Удалить выбросы"):
            df = remove_outliers(df)

        if st.checkbox("📈 Построить визуализацию"):
            visualize(df)

# === Запуск ===
if __name__ == "__main__":
    main()
