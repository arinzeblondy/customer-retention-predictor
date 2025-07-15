# customer-retention-predictor
Predict customer churn using LightGBM and visualize results with a sleek Streamlit app. Upload your data, get churn risk instantly, and download results. Built for e-commerce retention strategies with MLflow tracking and clean UI.

Project Overview

This project uses customer behavioral and transactional data to predict churn. It applies a LightGBM classifier and tracks performance using MLflow. The goal is to help e-commerce platforms retain more customers by identifying high-risk users early.

Dataset Description

Both datasets include:

| Feature           | Description                        |
| ----------------- | ---------------------------------- |
| CustomerID        | Unique identifier                  |
| Age               | Age of customer                    |
| Gender            | Male or Female                     |
| Tenure            | Duration with company (months)     |
| Usage Frequency   | Platform usage rate                |
| Support Calls     | Number of support requests         |
| Payment Delay     | Delayed payments count             |
| Subscription Type | e.g., Basic, Standard, Premium     |
| Contract Length   | Monthly, Quarterly, Annual         |
| Total Spend       | Lifetime spend amount              |
| Last Interaction  | Days since last platform use       |
| Churn             | 1 = churned, 0 = retained (target) |

Tech Stack

 Python (Pandas, NumPy, Seaborn, Matplotlib)
  LightGBM
  Scikit-learn
  MLflow
  Streamlit (for deployment)

Model Workflow

1. Data Cleaning: Remove missing entries
2. Encoding: Convert categorical features
3. Modeling: Train with LightGBM
4. Evaluation: ROC-AUC, classification report
5. Prediction: Add churn predictions to test set
6. Tracking: Use MLflow for experiment logging

Results
ROC AUC Score: \~0.85
Important Features: Payment Delay, Last Interaction, Usage Frequency

Visualizations

Churn distribution
Feature importance chart
 Boxplots for churn vs key drivers

Streamlit App (Coming Soon)

Upload customer data and get churn predictions instantly.
Visual dashboard of churn risk and feature drivers.


 `customer_churn_dataset-training-master.csv`
 `customer_churn_dataset-testing-master.csv`
 `churn_predictions.csv` (output)
  `customer_retention_ml.py` (main code)

 Future Improvements

 Hyperparameter tuning
 Add more behavioral features (e.g. clickstream)
 Send alerts via email API to customer success team

Contributions

Open to contributions! Submit a PR or raise issues for improvements.
Contact
Arinze Okechukwu
Machine Learning Engineer
[LinkedIn] (https://www.linkedin.com/in/arinze-okechukwu/)
[GitHub] (https://github.com/arinzeokechukwu)

Customer Retention Prediction App    http://localhost:8503/#87ae2447
