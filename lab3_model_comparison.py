import streamlit as st
import pandas as pd
import numpy as np
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io

st.set_page_config(page_title="ITD105 – Lab 3", layout="wide")

# ========== Classification Task ==========
def classification():
    st.header("Classification - Model Comparison")
    uploaded_file = st.file_uploader("Upload Classification Dataset (CSV)", type="csv", key="classification")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Dataset:")
        st.write(df.head())

        if df.isnull().sum().sum() > 0:
            st.warning("Missing values detected. Filling with column means.")
            df.fillna(df.mean(numeric_only=True), inplace=True)

        df = df.select_dtypes(include=[np.number])
        if df.shape[1] < 2:
            st.error("Insufficient numeric columns for classification.")
            return

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        st.subheader("Target Class Distribution")
        st.bar_chart(y.value_counts())

        target_type = type_of_target(y)
        if target_type not in ['binary', 'multiclass']:
            st.warning("Target appears continuous. Converting using median split.")
            try:
                y = pd.qcut(y, q=2, labels=[0, 1])
                st.success("Target converted to binary.")
            except Exception as e:
                st.error(f"Binning failed: {e}")
                return

        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier

        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'SVC': SVC(probability=True),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier()
        }

        metric_table = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in models.items():
            scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
            metric_table.append((name, scores.mean()))

        st.subheader("Classification Accuracy Table")
        result_df = pd.DataFrame(metric_table, columns=["Model", "Accuracy"])
        st.dataframe(result_df)

        st.subheader("Classification Accuracy Bar Chart")
        fig, ax = plt.subplots()
        sns.barplot(data=result_df, x='Model', y='Accuracy', ax=ax)
        st.pyplot(fig)

        best_model_name, best_score = max(metric_table, key=lambda x: x[1])
        best_model = models[best_model_name]
        best_model.fit(X, y)

        buffer = io.BytesIO()
        joblib.dump(best_model, buffer)
        buffer.seek(0)
        st.download_button(
            label=f"Download Best Classification Model ({best_model_name})",
            data=buffer,
            file_name=f"best_{best_model_name.lower().replace(' ', '_')}_model.joblib",
            mime="application/octet-stream"
        )

# ========== Regression Task ==========
def regression():
    st.header("Regression - Model Comparison")
    uploaded_file = st.file_uploader("Upload Regression Dataset (CSV)", type="csv", key="regression")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Dataset:")
        st.write(df.head())

        df = df.select_dtypes(include=[np.number])
        if df.isnull().sum().sum() > 0:
            df.fillna(df.mean(numeric_only=True), inplace=True)

        if "Next_Tmax" not in df.columns:
            st.error("Target column 'Next_Tmax' not found.")
            return

        X = df.drop(columns=["Next_Tmax"])
        y = df["Next_Tmax"]

        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor

        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'SVR': SVR(),
            'Decision Tree Regressor': DecisionTreeRegressor(),
            'Random Forest Regressor': RandomForestRegressor()
        }

        metric_table = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in models.items():
            try:
                mse_scores = -1 * cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
                r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
                metric_table.append((name, mse_scores.mean(), r2_scores.mean()))
            except ValueError as e:
                metric_table.append((name, np.nan, np.nan))
                st.warning(f"{name} failed: {e}")

        st.subheader("Regression Performance Table")
        result_df = pd.DataFrame(metric_table, columns=["Model", "MSE", "R²"])
        st.dataframe(result_df)

        st.subheader("Regression R² Bar Chart")
        fig, ax = plt.subplots()
        sns.barplot(data=result_df.dropna(), x='Model', y='R²', ax=ax)
        st.pyplot(fig)

        best_model_name, _, best_r2 = max(metric_table, key=lambda x: x[2] if not np.isnan(x[2]) else -np.inf)
        best_model = models[best_model_name]
        best_model.fit(X, y)

        buffer = io.BytesIO()
        joblib.dump(best_model, buffer)
        buffer.seek(0)
        st.download_button(
            label=f"Download Best Regression Model ({best_model_name})",
            data=buffer,
            file_name=f"best_{best_model_name.lower().replace(' ', '_')}_model.joblib",
            mime="application/octet-stream"
        )

# ========== Streamlit Layout ==========
def main():
    st.title("ITD105 – Lab 3: Model Comparison and Deployment")
    task = st.sidebar.radio("Select Task", ["Classification", "Regression"])
    if task == "Classification":
        classification()
    elif task == "Regression":
        regression()

if __name__ == "__main__":
    main()
