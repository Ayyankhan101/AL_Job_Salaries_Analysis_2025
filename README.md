# Data Science Salary Analysis and Prediction

## Introduction

This project analyzes a dataset of data science salaries to uncover insights into compensation trends and builds a predictive model to estimate salaries based on various factors. The analysis explores how factors like experience level, job title, company location, and work year influence salaries in the data science field.

The entire analysis and model development process is documented in the `Salary_Analysis_and_Prediction.ipynb` Jupyter Notebook.

## Dataset

The dataset used for this analysis is `salaries.csv`, which contains information about data science jobs and their corresponding salaries. The key columns in the dataset are:

*   `work_year`: The year the salary was paid.
*   `experience_level`: The experience level of the employee (e.g., Entry-level, Mid-level, Senior-level, Executive-level).
*   `employment_type`: The type of employment (e.g., Full-time, Part-time).
*   `job_title`: The specific role of the employee.
*   `salary`: The gross salary amount in the original currency.
*   `salary_currency`: The currency of the salary.
*   `salary_in_usd`: The salary converted to US Dollars (USD).
*   `employee_residence`: The employee's country of residence.
*   `remote_ratio`: The percentage of work done remotely.
*   `company_location`: The country where the company is located.
*   `company_size`: The size of the company (e.g., Small, Medium, Large).

## Installation

To run this project, you need Python and several libraries. You can install the necessary dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyterlab
```

## Usage

1.  Clone or download this repository to your local machine.
2.  Navigate to the project directory in your terminal.
3.  Launch Jupyter Lab by running:
    ```bash
    jupyter lab
    ```
4.  Open the `Salary_Analysis_and_Prediction.ipynb` notebook and run the cells to see the analysis and results.

## Project Workflow

The project follows these main steps:

1.  **Data Loading and Exploration**: The `salaries.csv` dataset is loaded, and an initial exploration is performed to understand its structure, data types, and basic statistics.
2.  **Data Cleaning and Feature Engineering**:
    *   Duplicate records are removed to ensure data quality.
    *   New features are created to aid the analysis and modeling process:
        *   `is_remote`: A binary flag indicating if the job is predominantly remote.
        *   `is_us`: A binary flag indicating if the company is located in the US.
        *   `job_family`: A simplified categorization of job titles (e.g., Data Scientist, Data Engineer, Machine Learning).
3.  **Exploratory Data Analysis (EDA)**:
    *   The distribution of salaries is visualized to understand its spread.
    *   Salaries are analyzed across different experience levels, job families, and company locations.
    *   The trend of median salaries over the years (2020-2025) is plotted.
4.  **Salary Prediction Model**:
    *   A machine learning model is built to predict `salary_in_usd`.
    *   **Features Used**: `experience_level`, `employment_type`, `job_family`, `remote_ratio`, `company_size`, and `is_us`.
    *   **Model**: A `RandomForestRegressor` is used within a scikit-learn `Pipeline` that handles preprocessing (scaling numerical features and one-hot encoding categorical features).
    *   **Hyperparameter Tuning**: `GridSearchCV` is employed to find the best combination of parameters for the model.
    *   **Evaluation**: The model's performance is assessed using Root Mean Squared Error (RMSE) and the R-squared (R2) score.

## Key Findings

*   **Salary Growth**: There has been a consistent upward trend in median data science salaries from 2020 to 2025.
*   **Experience Matters**: Salary is highly correlated with experience level, with executive-level roles commanding significantly higher compensation.
*   **Top Job Families**: The 'Machine Learning' and 'Data Scientist' job families tend to have the highest salaries.
*   **Geographic Influence**: The United States is the highest-paying location for data science professionals in this dataset.
*   **Model Performance**: The Random Forest model can predict salaries with a reasonable degree of accuracy, indicating that the selected features have predictive power.

## Future Improvements

*   Incorporate additional features that might influence salary (e.g., specific skills, technologies, or educational background).
*   Experiment with other regression models like Gradient Boosting or XGBoost to potentially improve prediction accuracy.
*   Gather more recent data to keep the analysis and model relevant.
