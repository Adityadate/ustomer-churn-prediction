# Customer Churn Prediction using Machine Learning

## 1. Project Goal
The goal of this project is to build and evaluate machine learning models to predict customer churn for a fictional telecommunications company. This helps the business identify at-risk customers and take proactive steps to retain them.

## 2. Dataset
The dataset is the "Telco Customer Churn" dataset from Kaggle. It contains customer demographics, account information, and services they've signed up for. The target variable is `Churn`.

## 3. Machine Learning Pipeline
The project follows a standard machine learning workflow:
1.  **Data Cleaning:** Handled missing values in `TotalCharges` and converted the column to a numeric type.
2.  **Preprocessing:**
    *   **Numerical Features:** Scaled using `StandardScaler`.
    *   **Categorical Features:** Encoded using `OneHotEncoder`.
    *   These steps were encapsulated in a `ColumnTransformer` and `Pipeline` for robustness.
3.  **Model Training:** Two models were trained for comparison:
    *   `LogisticRegression` (as a baseline)
    *   `RandomForestClassifier` (as a more powerful alternative)
4.  **Model Evaluation:** Models were evaluated on a held-out test set using key classification metrics: Accuracy, Precision, Recall, and F1-Score.

## 4. Results
*   **Logistic Regression:** Accuracy: 80.4%, Recall (for Churn): 0.55
*   **Random Forest:** Accuracy: 78.7%, Recall (for Churn): 0.49

**Conclusion:** While both models show decent performance, they highlight the trade-off between precision and recall. For this business problem, improving **recall** for the churn class is critical. Further work should focus on techniques like hyperparameter tuning or handling class imbalance to better identify customers at risk of churning.

## 5. How to Run
1.  Clone this repository.
2.  Create and activate a virtual environment.
3.  Install dependencies: `pip install -r requirements.txt`
4.  Run the `churn_prediction_model.ipynb` notebook.