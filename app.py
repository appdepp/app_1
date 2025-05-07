import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

        df.columns = df.columns.str.strip()  # Убираем пробелы в названиях колонок
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

# === 8. Визуализация данных ===
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

# === 9. Главная функция ===
def main():
    st.title("🧼 Очистка и анализ данных")

    df = load_data()
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

        # Кнопка для скачивания файла
        st.download_button(
            label="⬇️ Скачать как CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=filename,
            mime='text/csv'
        )

if __name__ == "__main__":
    main()