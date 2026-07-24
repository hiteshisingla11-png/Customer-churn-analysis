# Customer Churn Analysis 📊

A full end-to-end machine learning project to predict customer churn 
in a retail banking dataset of 10,000 customers.

## Project Overview
This project identifies why customers leave a bank and builds a 
predictive model to flag at-risk customers before they churn.

## Key Findings
- Overall churn rate: **20.4%** (1 in 5 customers)
- 🇩🇪 Germany has the highest churn rate at **32.4%** — double France and Spain
- Older customers (median age **45**) churn significantly more than retained customers (36)
- Inactive members churn at **26.9%** vs only **14.3%** for active members
- Best model: **Balanced Random Forest** (ROC-AUC: 0.865, Recall: 71%)

## Project Structure
| File | Description |
|---|---|
| `customer_churn.csv` | Original raw dataset (10,000 customers) |
| `customer_churn_cleaned.csv` | Cleaned and feature-engineered dataset |
| `app.py` | Streamlit web dashboard (live churn calculator + what-if simulator) |
| `churn_model.pkl` | Trained balanced random forest model |
| `Customer_Churn_Research_Paper.docx` | Full research paper with findings |
| `final_code_copy.ipynb` | Full Colab notebook with all code and outputs |

## Tech Stack
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

- Python, Pandas, Scikit-learn, Matplotlib, Seaborn
- Random Forest, Logistic Regression
- SHAP for model explainability
- Streamlit for interactive dashboard
- Google Colab for development

## Dashboard Features
- 🏠 **Overview** — churn distribution and key metrics
- 🎯 **Risk Calculator** — enter customer details, get live churn probability
- 📈 **Feature Importance** — which factors drive churn most
- 🔮 **What-If Simulator** — see how engagement changes affect churn risk

## How to Run the Dashboard
```bash
pip install streamlit scikit-learn pandas matplotlib seaborn joblib
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

## Model Performance
| Model | Recall | ROC-AUC |
|---|---|---|
| Logistic Regression | 19% | 0.773 |
| Random Forest | 43% | 0.864 |
| **Balanced Random Forest** | **71%** | **0.865** |
