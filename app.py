import streamlit as st
import pandas as pd
import os
from io import StringIO

def load_data():
    st.header("📥 Загрузка данных")

    files = [f for f in os.listdir() if f.endswith('.csv')]
    file_selected = st.selectbox("Выберите файл из директории:", files) if files else None
    uploaded_file = st.file_uploader("или загрузите файл с компьютера:", type=["csv", "xlsx"])

    df = None

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            st.success("✅ Файл загружен с компьютера")
        except Exception as e:
            st.error(f"❌ Ошибка при загрузке файла: {e}")
            return None
    elif file_selected:
        try:
            df = pd.read_csv(file_selected)
            st.success(f"✅ Файл '{file_selected}' успешно загружен")
        except Exception as e:
            st.error(f"❌ Ошибка при открытии файла: {e}")
            return None

    if df is not None:
        st.subheader("📊 Пример данных")
        st.dataframe(df.head())

        st.subheader("ℹ️ Информация о данных")
        buffer = StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

        return df

    return None