# Импортируем необходимые библиотеки
import streamlit as st  # Для создания веб-интерфейса
import pandas as pd     # Для работы с табличными данными
import numpy as np      # Для работы с массивами и числовыми операциями
import matplotlib.pyplot as plt  # Для построения графиков
import seaborn as sns   # Для улучшенных визуализаций на основе matplotlib
import os               # Для работы с файловой системой
from io import StringIO # Для отображения информации о DataFrame

# Функция загрузки данных из файла или через загрузчик
def load_data():
    st.header("📥 Загрузка данных")

    # Выбор способа загрузки
    method = st.radio("Выберите способ загрузки", [
        "Из списка файлов в папке",
        "Загрузить файл с компьютера"
    ])
    df = None  # Инициализируем переменную DataFrame

    # Загрузка из текущей директории
    if method == "Из списка файлов в папке":
        files = [f for f in os.listdir() if f.endswith(".csv")]
        if not files:
            st.warning("❌ В папке нет CSV-файлов.")
            return None
        file_selected = st.selectbox("Выберите файл", files)
        try:
            df = pd.read_csv(file_selected, on_bad_lines='skip')
        except Exception as e:
            st.error(f"❌ Ошибка при загрузке: {e}")
            return None

    # Загрузка с компьютера
    elif method == "Загрузить файл с компьютера":
        uploaded_file = st.file_uploader("Загрузите CSV-файл", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, on_bad_lines='skip')
            except Exception as e:
                st.error(f"❌ Ошибка при чтении файла: {e}")
                return None

    # Отображаем информацию о загруженных данных
    if df is not None:
        df = df.apply(lambda col: pd.to_numeric(col, errors='ignore') if col.dtypes == 'object' else col)
        st.success("✅ Данные успешно загружены")
        st.write("📊 Первые 5 строк данных")
        st.dataframe(df.head())
        st.write("ℹ️ Информация о DataFrame")

        # Выводим информацию о DataFrame как текст
        buffer = StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        return df

    return None

# Функция для показа пропущенных значений
def show_missing(df):
    st.subheader("📉 Анализ пропущенных значений")
    missing = df.isnull().sum()  # Количество пропусков в каждой колонке
    total_missing = missing.sum()  # Общее количество пропусков
    if total_missing == 0:
        st.success("✅ Нет пропущенных значений")
    else:
        st.warning("⚠️ Пропущенные значения:")
        st.dataframe(missing[missing > 0])
    return total_missing

# Функция ручного заполнения пропусков
def fill_missing(df):
    st.subheader("🧩 Заполнение пропусков вручную")
    col = st.selectbox("Выберите колонку", df.columns[df.isnull().any()])
    dtype = df[col].dtype  # Определяем тип данных выбранной колонки

    # Обработка числовых колонок
    if pd.api.types.is_numeric_dtype(dtype):
        method = st.selectbox("Метод заполнения", ["mean", "median", "dropna"])
        if method == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif method == "median":
            df[col] = df[col].fillna(df[col].median())
        elif method == "dropna":
            df = df.dropna(subset=[col])

    # Обработка строковых колонок
    elif pd.api.types.is_object_dtype(dtype):
        method = st.selectbox("Метод заполнения", ["mode", "Unknown", "dropna"])
        if method == "mode":
            df[col] = df[col].fillna(df[col].mode()[0])
        elif method == "Unknown":
            df[col] = df[col].fillna("Unknown")
        elif method == "dropna":
            df = df.dropna(subset=[col])

    # Обработка дат
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        method = st.selectbox("Метод заполнения", ["ffill", "bfill", "interpolate"])
        if method == "interpolate":
            df[col] = df[col].interpolate()
        else:
            df[col] = df[col].fillna(method=method)

    st.success(f"✅ Пропуски в колонке '{col}' обработаны")
    return df

# Функция автоматического заполнения пропусков
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

# Функция агрегации и построения графика
def aggregate_summary(df):
    st.subheader("📊 Агрегация и визуализация")

    # Выбор колонки для группировки и числовой колонки для агрегации
    group_col = st.selectbox("Колонка для группировки", df.columns)
    numeric_cols = df.select_dtypes(include=np.number).columns
    value_col = st.selectbox("Числовая колонка для агрегации", numeric_cols)

    # Выбор метода агрегации и типа графика
    agg_func = st.selectbox("Функция агрегации", ["mean", "sum", "count", "min", "max"])
    chart_type = st.selectbox("Тип графика", ["Гистограмма", "Диаграмма рассеяния", "Круговая диаграмма"])

    try:
        # Группировка и агрегация
        result = df.groupby(group_col)[value_col].agg(agg_func).reset_index()
        st.dataframe(result)

        # Построение графика
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

# Основная функция приложения
def main():
    st.title("🧼 Очистка и анализ данных")

    # Загрузка данных
    df = load_data()
    if df is None:
        return

    # Анализ и заполнение пропусков
    if show_missing(df) > 0:
        if st.checkbox("🧹 Удалить дубликаты"):
            duplicates = df.duplicated().sum()
            st.info(f"🔁 Найдено дубликатов: {duplicates}")
            if duplicates > 0 and st.button("Удалить дубликаты"):
                df = df.drop_duplicates()
                st.success("✅ Дубликаты удалены")
        if st.checkbox("📊 Показать describe()"):
            st.write(df.describe())
        if st.checkbox("🔧 Обработать пропущенные значения вручную"):
            df = fill_missing(df)
        if st.checkbox("🤖 Автоматически обработать все пропуски"):
            df = auto_fill_missing(df)

    # Агрегация и визуализация
    if st.checkbox("📈 Провести агрегацию и визуализацию"):
        aggregate_summary(df)
    if st.checkbox("🧠 Показать матрицу корреляции"):
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Сохранение обработанных данных
    if st.checkbox("💾 Сохранить обработанный DataFrame"):
        default_name = "cleaned_data.csv"
        filename = st.text_input("Введите имя файла для сохранения (с .csv)", value=default_name)

        if st.button("💾 Сохранить в рабочую директорию"):
            try:
                df.to_csv(filename, index=False)
                st.success(f"✅ Файл сохранён как '{filename}' в текущей директории")
            except Exception as e:
                st.error(f"❌ Ошибка при сохранении: {e}")

        # Скачивание обработанного файла
        st.download_button(
            label="⬇️ Скачать как CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=filename,
            mime='text/csv'
        )
        import io

        if st.checkbox("📥 Скачать как Excel (.xlsx)"):
            towrite = io.BytesIO()
            df.to_excel(towrite, index=False, sheet_name='Data')
            towrite.seek(0)
            st.download_button(
                label="⬇️ Скачать Excel-файл",
                data=towrite,
                file_name="cleaned_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# Запуск приложения
if __name__ == "__main__":
    main()