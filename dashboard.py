import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import warnings
import gzip

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

# Load data
@st.cache_data
def load_data():
    with gzip.open('salaries.csv.gz', 'rt') as f:
        df = pd.read_csv(f)
    return df

df = load_data()

# Data Cleaning and Feature Engineering
@st.cache_data
def clean_data(df):
    df = df.drop_duplicates()
    df['is_remote'] = df['remote_ratio'].apply(lambda x: 1 if x > 50 else 0)
    df['is_us'] = df['company_location'].apply(lambda x: 1 if x == 'US' else 0)

    def categorize_job(title):
        title = title.lower()
        if 'data scientist' in title:
            return 'Data Scientist'
        elif 'data engineer' in title:
            return 'Data Engineer'
        elif 'machine learning' in title or 'ml' in title:
            return 'Machine Learning'
        elif 'ai' in title:
            return 'AI'
        elif 'analyst' in title:
            return 'Analyst'
        elif 'manager' in title:
            return 'Manager'
        elif 'engineer' in title:
            return 'Engineer'
        elif 'software' in title:
            return 'Software Engineer'
        elif 'research' in title:
            return 'Research'
        else:
            return 'Other'

    df['job_family'] = df['job_title'].apply(categorize_job)
    return df

df_cleaned = clean_data(df.copy())

st.title("Data Science Salary Analysis and Prediction")

# Data Overview
st.header("Data Overview")
st.write(f"Dataset shape: {df.shape}")
st.write("### First 1000 rows:")
st.dataframe(df.head(1000))

st.write("### last 1000 rows:")
st.dataframe(df.tail(1000))

with st.expander("See full data profiling"):
    st.write("### Data types and missing values:")
    st.text(df.info())
    st.write("### Summary statistics:")
    st.dataframe(df.describe(include='all'))

# Exploratory Data Analysis
st.header("Exploratory Data Analysis")

with st.expander("Explore data distributions and trends"):
    # Distribution of Salaries
    st.subheader("Distribution of Salaries in USD")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(df_cleaned['salary_in_usd'], bins=50, kde=True, ax=ax, color='blue', stat='density', edgecolor='black')
    ax.set_title('Distribution of Salaries in USD', fontsize=16)
    ax.set_xlabel('Salary in USD', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    st.pyplot(fig)

    # Salary by Experience Level
    st.subheader("Salary Distribution by Experience Level")
    fig, ax = plt.subplots(figsize=(12, 6))
    order = ['EN', 'MI', 'SE', 'EX']
    sns.boxplot(x='experience_level', y='salary_in_usd', data=df_cleaned, order=order, ax=ax, palette='Set2', color='lightblue')
    ax.set_title('Salary Distribution by Experience Level', fontsize=16)
    ax.set_xlabel('Experience Level', fontsize=14)
    ax.set_ylabel('Salary in USD', fontsize=14)
    ax.set_xticklabels(['Entry', 'Mid', 'Senior', 'Executive'])
    st.pyplot(fig)

    # Salary by Job Family
    st.subheader("Salary Distribution by Job Family")
    fig, ax = plt.subplots(figsize=(14, 8))
    order = df_cleaned.groupby('job_family')['salary_in_usd'].median().sort_values(ascending=False).index
    sns.boxplot(x='job_family', y='salary_in_usd', data=df_cleaned, order=order, ax=ax, palette='Set2', color='lightgreen', width=0.8, fliersize=0)
    ax.set_title('Salary Distribution by Job Family', fontsize=16)
    ax.set_xlabel('Job Family', fontsize=14)
    ax.set_ylabel('Salary in USD', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    # Salary Trend by Year
    st.subheader("Median Salary Trend by Year")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='work_year', y='salary_in_usd', data=df_cleaned, estimator='median', errorbar=None, marker='o', ax=ax, color='purple', linewidth=2, markersize=8)
    ax.set_title('Median Salary Trend by Year', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Median Salary in USD', fontsize=14)
    ax.set_xticks(df_cleaned['work_year'].unique())
    st.pyplot(fig)

# Salary Prediction
st.header("Salary Prediction Model")

with st.expander("Predict your salary and see model performance"):
    # Model Training
    @st.cache_resource
    def train_model(df):
        features = ['experience_level', 'employment_type', 'job_family', 'remote_ratio', 'company_size', 'is_us']
        target = 'salary_in_usd'
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        categorical_features = ['experience_level', 'employment_type', 'job_family', 'company_size']
        numerical_features = ['remote_ratio', 'is_us']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', RandomForestRegressor(random_state=42))])

        param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [10, 20],
            'regressor__min_samples_leaf': [2, 4]
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        return best_model, X_test, y_test, y_pred, rmse, r2

    model, X_test, y_test, y_pred, rmse, r2 = train_model(df_cleaned)

    st.subheader("Model Performance")
    st.write(f"**Root Mean Squared Error (RMSE):** ${rmse:,.2f}")
    st.write(f"**R-squared (R2) Score:** {r2:.2f}")

    st.subheader("Actual vs. Predicted Salaries")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel('Actual Salary')
    ax.set_ylabel('Predicted Salary')
    ax.set_title('Actual vs. Predicted Salaries')
    st.pyplot(fig)

    st.subheader("Predict Your Salary")
    experience_level = st.selectbox("Experience Level", options=df_cleaned['experience_level'].unique())
    employment_type = st.selectbox("Employment Type", options=df_cleaned['employment_type'].unique())
    job_family = st.selectbox("Job Family", options=df_cleaned['job_family'].unique())
    remote_ratio = st.slider("Remote Ratio", 0, 100, 50)
    company_size = st.selectbox("Company Size", options=df_cleaned['company_size'].unique())
    is_us = st.checkbox("Company in US")

    if st.button("Predict Salary"):
        input_data = pd.DataFrame({
            'experience_level': [experience_level],
            'employment_type': [employment_type],
            'job_family': [job_family],
            'remote_ratio': [remote_ratio],
            'company_size': [company_size],
            'is_us': [1 if is_us else 0]
        })
        prediction = model.predict(input_data)
        st.success(f"Predicted Salary: ${prediction[0]:,.2f}")