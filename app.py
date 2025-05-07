import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO

def load_data():
    st.header("📥 Загрузка данных")

    method = st.radio("Выберите способ загрузки", ["Из списка файлов в папке", "Ввести путь вручную"])
    path = None

    if method == "Из списка файлов в папке":
        files = [f for f in os.listdir() if f.endswith(".csv")]
        if not files:
            st.warning("❌ В папке нет CSV-файлов.")
            return None
        file_selected = st.selectbox("Выберите файл", files)
        path = file_selected
    else:
        path = st.text_input("Введите путь к CSV-файлу:")

    if path:
        try:
            df = pd.read_csv(path)
            st.success(f"✅ Файл '{path}' успешно загружен")
            st.write("📊 Первые 5 строк данных", df.head())

            st.write("ℹ️ Информация о DataFrame")
            buffer = StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())

            return df
        except Exception as e:
            st.error(f"❌ Ошибка при загрузке: {e}")
            return None
    return None


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


def aggregate_summary(df):
    st.subheader("📊 Агрегация и визуализация")
    group_col = st.selectbox("Колонка для группировки", df.columns)
    numeric_cols = df.select_dtypes(include=np.number).columns
    value_col = st.selectbox("Числовая колонка для агрегации", numeric_cols)

    agg_func = st.selectbox("Функция агрегации", ["mean", "sum", "count", "min", "max"])
    chart_type = st.selectbox("Тип графика", ["Гистограмма", "Диаграмма рассеяния", "Круговая диаграмма"])

    try:
        result = df.groupby(group_col)[value_col].agg(agg_func).reset_index()
        st.dataframe(result)

        plt.figure(figsize=(10, 6))
        if chart_type == "Гистограмма":
            sns.barplot(x=group_col, y=value_col, data=result)
        elif chart_type == "Диаграмма рассеяния":
            sns.scatterplot(x=group_col, y=value_col, data=result)
        elif chart_type == "Круговая диаграмма":
            plt.pie(result[value_col], labels=result[group_col], autopct='%1.1f%%')

        plt.title(f"{agg_func.upper()} {value_col} по {group_col}")
        plt.xticks(rotation=45)

        fig = plt.gcf()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")


def main():
    st.title("🧼 Очистка и анализ данных")

    df = load_data()
    if df is None:
        return

    if show_missing(df) > 0:
        if st.checkbox("🔧 Обработать пропущенные значения вручную"):
            df = fill_missing(df)

        if st.checkbox("🤖 Автоматически обработать все пропуски"):
            df = auto_fill_missing(df)

    if st.checkbox("📈 Провести агрегацию и визуализацию"):
        aggregate_summary(df)

    if st.checkbox("💾 Сохранить обработанный DataFrame"):
        filename = st.text_input("Имя файла", "cleaned_data.csv")
        if st.button("Сохранить"):
            df.to_csv(filename, index=False)
            st.success(f"✅ Сохранено как {filename}")


if __name__ == "__main__":
    main()