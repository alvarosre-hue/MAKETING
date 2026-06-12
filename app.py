
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Bank Marketing EDA", layout="wide")

class DataAnalyzer:
    def __init__(self, df):
        self.df = df

    def estadisticas(self):
        return self.df.describe()

    def nulos(self):
        return self.df.isnull().sum()

    def numericas(self):
        return self.df.select_dtypes(include=np.number).columns

    def categoricas(self):
        return self.df.select_dtypes(include="object").columns

menu = st.sidebar.radio("Menú", ["Home", "Carga Dataset", "EDA"])

if menu == "Home":
    st.title("Bank Marketing EDA")
    st.write("Análisis Exploratorio de Datos para campañas de marketing bancario.")
    st.write("Autor: Álvaro Santiago Rivera Espinoza")
    st.write("Tecnologías: Python, Pandas, NumPy, Streamlit, Matplotlib y Seaborn")

elif menu == "Carga Dataset":
    archivo = st.file_uploader("Seleccione el archivo CSV", type=["csv"])

    if archivo is not None:
        df = pd.read_csv(archivo, sep=";")
        st.success("Dataset cargado correctamente")
        st.dataframe(df.head())
        st.write(f"Filas: {df.shape[0]}")
        st.write(f"Columnas: {df.shape[1]}")

elif menu == "EDA":
    archivo = st.file_uploader("Seleccione el archivo CSV", type=["csv"])

    if archivo is not None:
        df = pd.read_csv(archivo, sep=";")
        analyzer = DataAnalyzer(df)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Items 1-2", "Items 3-4", "Items 5-6", "Items 7-8", "Items 9-10"]
        )

        with tab1:
            st.header("Información General")
            st.write(df.dtypes)
            st.write(analyzer.nulos())

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Variables Numéricas")
                st.write(list(analyzer.numericas()))
            with col2:
                st.subheader("Variables Categóricas")
                st.write(list(analyzer.categoricas()))

        with tab2:
            st.header("Estadísticas Descriptivas")
            st.dataframe(analyzer.estadisticas())
            st.bar_chart(analyzer.nulos())

        with tab3:
            variable = st.selectbox("Variable numérica", analyzer.numericas())
            fig, ax = plt.subplots()
            sns.histplot(df[variable], kde=True, ax=ax)
            st.pyplot(fig)

            categoria = st.selectbox("Variable categórica", analyzer.categoricas())
            st.bar_chart(df[categoria].value_counts())

        with tab4:
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x="y", y="age", ax=ax)
            st.pyplot(fig)

            fig, ax = plt.subplots()
            sns.countplot(data=df, x="contact", hue="y", ax=ax)
            st.pyplot(fig)

        with tab5:
            columnas = st.multiselect("Seleccione columnas", df.columns)

            if columnas:
                st.dataframe(df[columnas].head(20))

            edad = st.slider("Edad máxima", 18, 100, 50)
            st.dataframe(df[df["age"] <= edad].head(20))

            if st.checkbox("Mostrar estadísticas"):
                st.dataframe(df.describe())

            st.success("""
            Hallazgos clave:
            1. La mayoría de clientes no acepta la campaña.
            2. La duración de la llamada influye en la aceptación.
            3. Existen diferencias entre segmentos.
            4. El canal de contacto afecta los resultados.
            5. Las campañas previas aportan información útil.
            """)
