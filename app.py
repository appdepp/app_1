# Import necessary libraries
import streamlit as st  # For creating the web interface
import pandas as pd     # For working with tabular data
import numpy as np      # For working with arrays and numerical operations
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns   # For enhanced visualizations based on matplotlib
import os               # For file system operations
from io import StringIO # For displaying information about DataFrame

# Function to load data from a file or using a file uploader
def load_data():
    st.header("📥 Data Upload")

    # Choose the upload method
    method = st.radio("Select the upload method", [
        "From file list in folder",
        "Upload file from computer"
    ])
    df = None  # Initialize the DataFrame variable

    # Loading from the current directory
    if method == "From file list in folder":
        files = [f for f in os.listdir() if f.endswith(".csv")]
        if not files:
            st.warning("❌ No CSV files in the folder.")
            return None
        file_selected = st.selectbox("Select a file", files)
        try:
            df = pd.read_csv(file_selected, on_bad_lines='skip')
        except Exception as e:
            st.error(f"❌ Error while loading: {e}")
            return None

    # Uploading from computer
    elif method == "Upload file from computer":
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, on_bad_lines='skip')
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
                return None

    # Display data information
    if df is not None:
        df = df.apply(lambda col: pd.to_numeric(col, errors='ignore') if col.dtypes == 'object' else col)
        st.success("✅ Data loaded successfully")
        st.write("📊 First 5 rows of the data")
        st.dataframe(df.head())
        st.write("ℹ️ DataFrame Information")

        # Display DataFrame info as text
        buffer = StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        return df

    return None

# Function to show missing values
def show_missing(df):
    st.subheader("📉 Missing Values Analysis")
    missing = df.isnull().sum()  # Count of missing values in each column
    total_missing = missing.sum()  # Total missing values
    if total_missing == 0:
        st.success("✅ No missing values")
    else:
        st.warning("⚠️ Missing values:")
        st.dataframe(missing[missing > 0])
    return total_missing

# Function to manually fill missing values
def fill_missing(df):
    st.subheader("🧩 Manual Missing Value Filling")
    col = st.selectbox("Select column", df.columns[df.isnull().any()])
    dtype = df[col].dtype  # Determine the data type of the selected column

    # Handling numerical columns
    if pd.api.types.is_numeric_dtype(dtype):
        method = st.selectbox("Filling method", ["mean", "median", "dropna"])
        if method == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif method == "median":
            df[col] = df[col].fillna(df[col].median())
        elif method == "dropna":
            df = df.dropna(subset=[col])

    # Handling string columns
    elif pd.api.types.is_object_dtype(dtype):
        method = st.selectbox("Filling method", ["mode", "Unknown", "dropna"])
        if method == "mode":
            df[col] = df[col].fillna(df[col].mode()[0])
        elif method == "Unknown":
            df[col] = df[col].fillna("Unknown")
        elif method == "dropna":
            df = df.dropna(subset=[col])

    # Handling date columns
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        method = st.selectbox("Filling method", ["ffill", "bfill", "interpolate"])
        if method == "interpolate":
            df[col] = df[col].interpolate()
        else:
            df[col] = df[col].fillna(method=method)

    st.success(f"✅ Missing values in column '{col}' have been handled")
    return df

# Function for automatic filling of missing values
def auto_fill_missing(df):
    st.subheader("⚙️ Auto-filling All Missing Values")
    for col in df.columns[df.isnull().any()]:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            df[col] = df[col].fillna(df[col].mean())
        elif pd.api.types.is_object_dtype(dtype):
            df[col] = df[col].fillna("Unknown")
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            df[col] = df[col].fillna(method='ffill')
    st.success("✅ All missing values have been processed automatically")
    return df

# Function for aggregation and plotting
def aggregate_summary(df):
    st.subheader("📊 Aggregation and Visualization")

    # Select column for grouping and numeric column for aggregation
    group_col = st.selectbox("Column for grouping", df.columns)
    numeric_cols = df.select_dtypes(include=np.number).columns
    value_col = st.selectbox("Numeric column for aggregation", numeric_cols)

    # Select aggregation function and chart type
    agg_func = st.selectbox("Aggregation function", ["mean", "sum", "count", "min", "max"])
    chart_type = st.selectbox("Chart type", ["Histogram", "Scatter plot", "Pie chart"])

    try:
        # Grouping and aggregation
        result = df.groupby(group_col)[value_col].agg(agg_func).reset_index()
        st.dataframe(result)

        # Plotting
        plt.figure(figsize=(10, 6))
        if chart_type == "Histogram":
            sns.barplot(x=group_col, y=value_col, data=result)
        elif chart_type == "Scatter plot":
            sns.scatterplot(x=group_col, y=value_col, data=result)
        elif chart_type == "Pie chart":
            plt.pie(result[value_col], labels=result[group_col], autopct='%1.1f%%')

        plt.title(f"{agg_func.upper()} {value_col} by {group_col}")
        plt.xticks(rotation=45)
        fig = plt.gcf()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Error: {e}")

# Function to clear files from the current directory
def clear_data_folder():
    st.subheader("🧹 Clear Data Folder")
    if st.button("Clear all CSV files in the folder"):
        try:
            files = [f for f in os.listdir() if f.endswith(".csv")]
            if files:
                for file in files:
                    os.remove(file)
                st.success("✅ All CSV files have been deleted from the folder")
            else:
                st.warning("❌ No CSV files found in the folder to delete.")
        except Exception as e:
            st.error(f"❌ Error deleting files: {e}")

# Main function for the application
def main():
    st.title("🧼 Data Cleaning and Analysis")

    # Load data
    df = load_data()
    if df is None:
        return

    # Analyze and fill missing values
    if show_missing(df) > 0:
        if st.checkbox("🧹 Remove duplicates"):
            duplicates = df.duplicated().sum()
            st.info(f"🔁 Duplicates found: {duplicates}")
            if duplicates > 0 and st.button("Remove duplicates"):
                df = df.drop_duplicates()
                st.success("✅ Duplicates removed")
        if st.checkbox("📊 Show describe()"):
            st.write(df.describe())
        if st.checkbox("🔧 Handle missing values manually"):
            df = fill_missing(df)
        if st.checkbox("🤖 Automatically handle all missing values"):
            df = auto_fill_missing(df)

    # Aggregation and visualization
    if st.checkbox("📈 Perform aggregation and visualization"):
        aggregate_summary(df)
    if st.checkbox("🧠 Show correlation matrix"):
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Clear data folder
    clear_data_folder()

    # Save processed data
    if st.checkbox("💾 Save processed DataFrame"):
        default_name = "cleaned_data.csv"
        filename = st.text_input("Enter file name for saving (with .csv)", value=default_name)

        if st.button("💾 Save to current directory"):
            try:
                df.to_csv(filename, index=False)
                st.success(f"✅ File saved as '{filename}' in the current directory")
            except Exception as e:
                st.error(f"❌ Error saving file: {e}")

        # Download the processed file
        st.download_button(
            label="⬇️ Download as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=filename,
            mime='text/csv'
        )
        import io

        if st.checkbox("📥 Download as Excel (.xlsx)"):
            towrite = io.BytesIO()
            df.to_excel(towrite, index=False, sheet_name='Data')
            towrite.seek(0)
            st.download_button(
                label="⬇️ Download Excel file",
                data=towrite,
                file_name="cleaned_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# Run the application
if __name__ == "__main__":
    main()