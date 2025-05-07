import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO




def try_read_csv(file):
    """Попытка прочитать CSV с несколькими кодировками"""
    encodings = ['utf-8', 'utf-16', 'cp1251', 'ISO-8859-1']
    for enc in encodings:
        try:
            return pd.read_csv(file, encoding=enc)
        except Exception:
            continue
    raise ValueError("❌ Не удалось прочитать CSV с поддерживаемыми кодировками")

def load_data():
    st.header("📥 Загрузка данных")
    method = st.radio("Выберите способ загрузки", ["Из списка файлов", "Загрузить с компьютера"])
    df = None

    if method == "Из списка файлов":
        files = [f for f in os.listdir() if f.endswith(".csv")]
        if not files:
            st.warning("❌ Нет CSV-файлов в папке")
            return None
        file_selected = st.selectbox("Выберите файл", files)
        try:
            df = try_read_csv(file_selected)
        except Exception as e:
            st.error(f"❌ Ошибка загрузки: {e}")
            return None
    else:
        uploaded_file = st.file_uploader("Загрузите CSV-файл", type="csv")
        if uploaded_file:
            try:
                df = try_read_csv(uploaded_file)
            except Exception as e:
                st.error(f"❌ Ошибка при чтении: {e}")
                return None

    if df is not None:
        st.success("✅ Данные загружены")
        st.write("👀 Предварительный просмотр:")
        st.dataframe(df.head())

        buffer = StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

        return df

    return None

def show_missing(df):
    st.subheader("📉 Пропущенные значения")
    missing = df.isnull().sum()
    total = missing.sum()
    if total == 0:
        st.success("✅ Нет пропущенных значений")
    else:
        st.warning(f"⚠️ Обнаружено {total} пропусков")
        st.dataframe(missing[missing > 0])
    return total

def fill_missing(df):
    st.subheader("🛠 Ручное заполнение пропусков")
    col = st.selectbox("Выберите колонку с пропусками", df.columns[df.isnull().any()])
    dtype = df[col].dtype

    if pd.api.types.is_numeric_dtype(dtype):
        method = st.selectbox("Метод", ["mean", "median", "dropna"])
        if method == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif method == "median":
            df[col] = df[col].fillna(df[col].median())
        elif method == "dropna":
            df = df.dropna(subset=[col])

    elif pd.api.types.is_object_dtype(dtype):
        method = st.selectbox("Метод", ["mode", "Unknown", "dropna"])
        if method == "mode":
            df[col] = df[col].fillna(df[col].mode()[0])
        elif method == "Unknown":
            df[col] = df[col].fillna("Unknown")
        elif method == "dropna":
            df = df.dropna(subset=[col])

    elif pd.api.types.is_datetime64_any_dtype(dtype):
        method = st.selectbox("Метод", ["ffill", "bfill", "interpolate"])
        if method == "interpolate":
            df[col] = df[col].interpolate()
        else:
            df[col] = df[col].fillna(method=method)

    st.success(f"✅ Пропуски в '{col}' обработаны")
    return df

def auto_fill_missing(df):
    st.subheader("🤖 Автоматическая обработка пропусков")
    for col in df.columns[df.isnull().any()]:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            df[col] = df[col].fillna(df[col].mean())
        elif pd.api.types.is_object_dtype(dtype):
            df[col] = df[col].fillna("Unknown")
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            df[col] = df[col].fillna(method='ffill')
    st.success("✅ Все пропуски обработаны")
    return df

def remove_duplicates(df):
    st.subheader("🧹 Удаление дубликатов")
    duplicates = df.duplicated().sum()
    if duplicates == 0:
        st.info("✅ Дубликаты не найдены")
    else:
        st.warning(f"⚠️ Найдено {duplicates} дубликатов")
        if st.button("Удалить дубликаты"):
            df = df.drop_duplicates()
            st.success("✅ Дубликаты удалены")
    return df

def aggregate_summary(df):
    st.subheader("📊 Агрегация и графики")
    group_col = st.selectbox("Группировать по", df.columns)
    numeric_cols = df.select_dtypes(include=np.number).columns
    value_col = st.selectbox("Числовая колонка", numeric_cols)

    agg_func = st.selectbox("Функция", ["mean", "sum", "count", "min", "max"])
    chart_type = st.selectbox("Тип графика", ["Гистограмма", "Диаграмма рассеяния", "Круговая"])

    try:
        result = df.groupby(group_col)[value_col].agg(agg_func).reset_index()
        st.dataframe(result)

        plt.figure(figsize=(10, 6))
        if chart_type == "Гистограмма":
            sns.barplot(x=group_col, y=value_col, data=result)
        elif chart_type == "Диаграмма рассеяния":
            sns.scatterplot(x=group_col, y=value_col, data=result)
        elif chart_type == "Круговая":
            plt.pie(result[value_col], labels=result[group_col], autopct='%1.1f%%')

        plt.title(f"{agg_func.upper()} {value_col} по {group_col}")
        plt.xticks(rotation=45)
        st.pyplot(plt.gcf())
    except Exception as e:
        st.error(f"❌ Ошибка визуализации: {e}")

def main():
    st.set_page_config(page_title="Data Cleaner", layout="wide")
    st.title("🧼 Data Cleaner: Очистка и анализ CSV")

    df = load_data()
    if df is None:
        return

    if show_missing(df) > 0:
        if st.checkbox("🔧 Ручная обработка пропусков"):
            df = fill_missing(df)
        if st.checkbox("⚙️ Автоматическая обработка"):
            df = auto_fill_missing(df)

    df = remove_duplicates(df)

    if st.checkbox("📈 Визуализация и агрегирование"):
        aggregate_summary(df)

    if st.checkbox("💾 Сохранение результата"):
        filename = st.text_input("Имя файла:", "cleaned_data.csv")
        if filename:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Скачать CSV", csv, file_name=filename, mime="text/csv")

if __name__ == "__main__":
    main()