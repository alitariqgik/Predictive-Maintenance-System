# Predictive Maintenance System 

An end-to-end Machine Learning pipeline designed to anticipate machine failures before they occur. This project was developed as part of the DS211 Data Science course and focuses on identifying early warning signs of equipment degradation using sensor data.

## Project Overview

In industrial settings, machine failures are highly imbalanced events (healthy operations vastly outnumber failures). This project tackles that challenge by employing cost-sensitive learning and Synthetic Minority Over-sampling Technique (SMOTE). It trains and evaluates multiple classification models to optimize for **F1-Score, Precision, and Recall**, ensuring that potential failures are reliably caught without generating excessive false alarms.

### Key Features
- **Automated Data Generation:** Script to synthesize realistic machine sensor and operational data.
- **Robust Preprocessing:** Handles data scaling, encoding, and SMOTE for class imbalance.
- **Multi-Model Training:** Compares Logistic Regression, Decision Trees, Random Forest, XGBoost, and SVM.
- **Cost-Sensitive Evaluation:** Implements custom probability thresholding and class weights to penalize missed failures.
- **Modular Architecture:** Built with clean, production-ready software engineering principles.

## Repository Structure

```text
├── data/                   # Generated datasets (ignored in git)
├── models/                 # Saved trained models (.pkl files)
├── results/                # Output evaluation metrics and reports
├── src/                    # Core pipeline modules
│   ├── data_loader.py      # Data ingestion logic
│   ├── preprocessing.py    # SMOTE, scaling, and feature engineering
│   ├── train.py            # Model training algorithms
│   ├── evaluate.py         # Metrics, scoring, and threshold optimization
│   └── predict.py          # Inference pipeline for new data
├── generate_data.py        # Script to create the synthetic dataset
├── main.py                 # Main execution script to run the pipeline
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation
```

## How to Run the Project

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/predictive-maintenance-ds211.git
cd predictive-maintenance-ds211
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the full pipeline**
The main script will automatically generate data, preprocess it, train the models, evaluate them, and run a sample prediction.
```bash
python main.py
```

## Evaluation & Results
Evaluation reports, including precision-recall trade-offs and confusion matrices, are automatically generated and saved to the `/results` directory after running the pipeline. The system evaluates models to ensure it leans toward high recall (catching failures) while maintaining a respectable precision.

## Technologies Used
- **Python 3.x**
- **Scikit-Learn** & **XGBoost** (Machine Learning)
- **Imbalanced-Learn** (SMOTE)
- **Pandas & NumPy** (Data Manipulation)
