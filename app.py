import os
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
from scipy.stats import zscore

# === 1. Загрузка данных ===
def load_data(file_selected=None, uploaded_file=None):
    st.header("📥 Загрузка данных")

    df = None

    if file_selected:
        try:
            df = pd.read_csv(file_selected)
        except Exception as e:
            st.error(f"❌ Ошибка при загрузке: {e}")
            return None

    elif uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Ошибка при чтении файла: {e}")
            return None

    if df is not None:
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

# === 2. Обработка пропусков ===
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

def fill_missing(df):
    st.subheader("🧩 Заполнение пропусков вручную")
    col = st.selectbox("Выберите колонку", df.columns[df.isnull().any()])
    dtype = df[col].dtype

    if pd.api.types.is_numeric_dtype(dtype):
        method = st.selectbox("Метод заполнения", ["mean", "median", "dropna"])
        if method == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif method == "median":
            df[col] = df[col].fillna(df[col].median())
        elif method == "dropna":
            df = df.dropna(subset=[col])
    elif pd.api.types.is_object_dtype(dtype):
        method = st.selectbox("Метод заполнения", ["mode", "Unknown", "dropna"])
        if method == "mode":
            df[col] = df[col].fillna(df[col].mode()[0])
        elif method == "Unknown":
            df[col] = df[col].fillna("Unknown")
        elif method == "dropna":
            df = df.dropna(subset=[col])
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        method = st.selectbox("Метод заполнения", ["ffill", "bfill", "interpolate"])
        if method == "interpolate":
            df[col] = df[col].interpolate()
        else:
            df[col] = df[col].fillna(method=method)

    st.success(f"✅ Пропуски в колонке '{col}' обработаны")
    return df

def auto_fill_missing(df):
    st.subheader("⚙️ Автоматическое заполнение всех пропусков")
    for col in df.columns[df.isnull().any()]:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            df[col] = df[col].fillna(df[col].mean())
        elif pd.api.types.is_object_dtype(dtype):
            df[col] = df[col].fillna("Unknown")
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            df[col] = df[col].fillna(method='ffill')
    st.success("✅ Все пропуски обработаны автоматически")
    return df

# === 3. Удаление дубликатов ===
def remove_duplicates(df):
    st.subheader("🧹 Удаление дубликатов")
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    st.success(f"✅ Удалено {before - after} дубликатов")
    return df

# === 4. Удаление выбросов ===
def remove_outliers(df):
    st.subheader("📏 Удаление выбросов")
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) < 1:
        st.warning("⚠️ Нет числовых колонок для удаления выбросов.")
        return df

    for col in numeric_cols:
        st.write(f"Обработка выбросов для колонки '{col}'")
        z_scores = np.abs(zscore(df[col].dropna()))
        df = df[(z_scores < 3)]

    st.success("✅ Все выбросы удалены.")
    return df

# === 5. Визуализация данных ===
def visualize(df):
    st.subheader("📊 Визуализация")
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) < 2:
        st.warning("⚠️ Недостаточно числовых колонок для визуализации.")
        return

    x_col = st.selectbox("Ось X", numeric_cols, key="x")
    y_col = st.selectbox("Ось Y", numeric_cols, key="y")

    # Выбор типа графика
    plot_type = st.selectbox("Выберите тип графика", ["Точечная диаграмма", "Гистограмма", "Линейный график"])

    plt.figure(figsize=(10, 5))

    if plot_type == "Точечная диаграмма":
        sns.scatterplot(data=df, x=x_col, y=y_col)
        plt.title(f"Точечная диаграмма: {x_col} vs {y_col}")
    elif plot_type == "Гистограмма":
        sns.histplot(df[x_col], kde=True)
        plt.title(f"Гистограмма для: {x_col}")
    elif plot_type == "Линейный график":
        sns.lineplot(data=df, x=x_col, y=y_col)
        plt.title(f"Линейный график: {x_col} vs {y_col}")

    st.pyplot(plt)

# === 6. Главная функция ===
def main():
    st.title("🧼 Очистка и анализ данных")

    # Кнопка для обновления списка файлов
    files = [f for f in os.listdir() if f.endswith(".csv")]
    files_placeholder = st.empty()
    if st.button("🔄 Обновить список файлов"):
        if files:
            files_placeholder.selectbox("Выберите файл", files)
        else:
            st.warning("❌ Нет доступных файлов")

    # Загрузка данных
    file_selected = st.selectbox("Выберите файл", files) if files else None
    uploaded_file = st.file_uploader("Загрузите файл с компьютера", type=["csv", "xlsx"])

    df = load_data(file_selected=file_selected, uploaded_file=uploaded_file)
    if df is None:
        return

    st.subheader("📊 Обзор данных")
    st.dataframe(df.head())

    # Обработка пропущенных значений
    total_missing = show_missing(df)
    if total_missing > 0:
        st.markdown("### 🔍 Обнаружены пропущенные значения")
        if st.checkbox("🔧 Обработать вручную"):
            df = fill_missing(df)

        if st.checkbox("🤖 Автоматически заполнить пропуски"):
            df = auto_fill_missing(df)

    # Удаление дубликатов
    if st.checkbox("🧹 Удалить дубликаты"):
        df = remove_duplicates(df)

    # Удаление выбросов
    if st.checkbox("📏 Удалить выбросы"):
        df = remove_outliers(df)

    # Агрегация и визуализация
    if st.checkbox("📈 Провести агрегацию и визуализацию"):
        visualize(df)

    # Сохранение и скачивание
    if st.checkbox("💾 Сохранить обработанный DataFrame"):
        default_name = "cleaned_data.csv"
        filename = st.text_input("Введите имя файла для сохранения (с .csv)", value=default_name)

        if st.button("💾 Сохранить в рабочую директорию"):
            try:
                df.to_csv(filename, index=False)
                st.success(f"✅ Файл сохранён как '{filename}' в текущей директории")
            except Exception as e:
                st.error(f"❌ Ошибка при сохранении: {e}")

        # Кнопка для скачивания
        st.download_button(
            label="⬇️ Скачать как CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=filename,
            mime='text/csv'
        )

if __name__ == "__main__":
    main()