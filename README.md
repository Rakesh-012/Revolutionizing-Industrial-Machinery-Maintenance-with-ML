# Revolutionizing Industrial Machinery Maintenance with ML

A machine learning project for **predictive maintenance** in industrial machinery. By analyzing sensor data, this system predicts equipment operational states across multiple failure types, reducing downtime, optimizing maintenance schedules, and improving efficiency.

The project includes a **Flask web interface** and a **chatbot for feature explanations**, making it user-friendly for industrial applications.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Repository Structure](#repository-structure)
* [Dataset Overview](#dataset-overview)
* [Technologies Used](#technologies-used)
* [Getting Started](#getting-started)
* [Running the Application](#running-the-application)
* [Web Interface Preview](#web-interface-preview)
* [Model Prediction Output](#model-prediction-output)
* [Model Training and Evaluation](#model-training-and-evaluation)
* [Challenges and Solutions](#challenges-and-solutions)
* [Conclusion and Future Scope](#conclusion-and-future-scope)
* [Acknowledgments](#acknowledgments)

---

## Project Overview

The project predicts machinery failures based on sensor and operational data. It classifies machinery states into:

* **No Failure**
* **Power Failure**
* **Tool Wear Failure**
* **Heat Dissipation Failure**
* **Overstrain Failure**
* **Random Failures**

Workflow:

* Data preprocessing & feature engineering
* Handling null values and data integrity checks
* Multiclass model training & evaluation
* Hyperparameter tuning using GridSearchCV
* Deployment with **Flask** and **chatbot integration**

---

## Repository Structure

```
├── app.py
├── Predictive Maintenance multiclass classification.ipynb
├── predictive_maintenance.csv
├── model.pkl
├── train model.py
├── templates/
      └── index.html
├── shap_background_data.csv
└── README.md
```
---

## Dataset Overview

**Features:**

* Air Temperature (K)
* Rotational Speed (RPM)
* Torque (Nm)
* Tool Wear (min)
* Process Temperature (K)
* Other operational parameters

**Target Classes:**

* No Failure
* Power Failure
* Tool Wear Failure
* Heat Dissipation Failure
* Overstrain Failure
* Random Failures

**Preprocessing Steps:**

* Removed unnecessary columns
* Checked for null values
* Verified data types

---

## Technologies Used

* **Programming Language:** Python 3.x
* **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, XGBoost, CatBoost, LightGBM
* **Web Deployment:** Flask
* **Model Serialization:** pickle
* **Handling Imbalanced Classes:** SMOTETomek

---

## Getting Started

### Prerequisites

* Python 3.x
* pip

### Installation

```bash
git clone https://github.com/your-username/PDM.git
cd PDM
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

---

## Running the Application

```bash
python app.py
```

Open browser: `http://127.0.0.1:5000/`

---

## Web Interface Preview

**Input & Prediction Page / Chatbot for Feature Understanding**  
<img width="2000" height="1125" alt="image" src="https://github.com/user-attachments/assets/4c8612e4-11fc-4a27-80d7-90ae8c81853f" />

**Model Prediction Output**  
<img width="2000" height="1125" alt="image" src="https://github.com/user-attachments/assets/9d62f68a-d993-4013-a43d-9717ba1ea369" />

---

## Model Prediction Output

Predicted classes:

* No Failure
* Power Failure
* Tool Wear Failure
* Heat Dissipation Failure
* Overstrain Failure
* Random Failures

---

## Model Training and Evaluation

| Model         | Accuracy (%) |
| ------------- | ------------ |
| Random Forest | 95.88        |
| XGBoost       | 95.06        |
| CatBoost      | 95.18        |
| LightGBM      | 92.94        |

* Metrics: Accuracy, Precision, Recall, F1-score
* Feature importance analysis using SHAP
* Hyperparameter tuning with GridSearchCV
* Imbalanced classes handled via SMOTETomek

---

## Challenges and Solutions

* **Imbalanced Classes:** Used SMOTETomek to improve recall for minority failure types
* **Optimizing Models:** GridSearchCV tuned hyperparameters for best predictive performance

---

## Conclusion and Future Scope

**Conclusion:**

* ML models accurately classify machinery failure types
* Flask web interface and chatbot enhance usability
* Reduces unexpected downtime and optimizes maintenance

**Future Scope:**

* Real-time sensor data integration
* More machinery types and additional failure modes
* Deep learning models for complex datasets
* Cloud-based API deployment for industrial integration

---

## Acknowledgments

* Inspired by the need for **proactive industrial maintenance**
* Dataset sourced from [Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification/code)
