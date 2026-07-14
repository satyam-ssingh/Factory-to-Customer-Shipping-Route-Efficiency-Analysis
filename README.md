# 🍬 Factory-to-Customer Shipping Route Efficiency Analysis

A complete Data Analytics, Business Intelligence, and Machine Learning project built on the Nassau Candy Distributor dataset to evaluate and optimize factory-to-customer shipping operations across the United States.

This project follows an end-to-end analytics workflow starting from raw data cleaning and feature engineering to advanced exploratory analysis, business intelligence reporting, interactive dashboard development, and machine learning model training.

---

# 🌐 Live Dashboard

🚀 Explore the live interactive dashboard here:

**👉 https://factsatyamp-eivrfhwlvsznxv2zlpkjhu.streamlit.app/**

---

# 📖 Project Overview

Efficient logistics and transportation management are critical for large-scale distributors. Shipping delays, inefficient routes, and regional bottlenecks can significantly impact customer satisfaction, operational costs, and overall business performance.

This project was developed to analyze shipping operations for Nassau Candy Distributor and answer key business questions such as:

* Which shipping routes are the most efficient?
* Which routes experience frequent delays?
* Which states and regions have poor shipping performance?
* How does ship mode affect delivery speed and profitability?
* Which factories perform best operationally?
* Can shipment delays be predicted using machine learning?

The project converts raw shipment records into meaningful business insights and predictive analytics that support data-driven logistics decisions.

---

# 🎯 Project Objectives

### Business Objectives

* Improve shipping efficiency
* Identify logistics bottlenecks
* Reduce shipment delays
* Compare performance across factories
* Evaluate regional and state-level performance
* Analyze shipping mode effectiveness
* Support operational decision-making

### Technical Objectives

* Perform large-scale data cleaning
* Engineer business-focused features
* Conduct exploratory data analysis
* Build interactive business dashboards
* Develop predictive machine learning models
* Generate executive-level reports and recommendations

---

# 🏗 Project Architecture

```text
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Business Analytics
 ├─ Ship Mode Analysis
 ├─ State Analysis
 ├─ Region Analysis
 └─ Factory Analysis
      │
      ▼
Interactive Dashboard
      │
      ▼
Machine Learning
      │
      ▼
Business Recommendations
```

---

# 📂 Project Structure

## 🗂 Entity Relationship Diagram
```text
Factory-to-Customer-Shipping-Route-Efficiency-Analysis/

├── 01_data_cleaning.py
├── 02_feature_engineering.py
├── 03_exploratory_data_analysis.py
├── 04_ship_mode_analysis.py
├── 05_state_analysis.py
├── 06_region_analysis.py
├── 07_factory_analysis.py
├── 08_streamlit_dashboard.py
├── 09_ml_training_and_evaluation.py

├── Nassau_Candy_Distributor.csv

├── cleaned_nassau_candy.csv
├── featured_nassau_candy.csv

├── EDA_Charts/
├── EDA_Summaries/

├── ML_Charts/
├── ML_Reports/
├── ML_Models/

└── README.md
```

---

# 🔍 Phase 1 — Data Cleaning

The first stage focuses on transforming raw shipment data into a reliable analytical dataset.

### Tasks Performed

* Dataset inspection
* Missing value analysis
* Duplicate record detection
* Date validation and conversion
* Shipping lead time calculation
* Invalid shipment record handling
* Data consistency checks

### Output

```text
cleaned_nassau_candy.csv
```

---

# ⚙️ Phase 2 — Feature Engineering

Business-oriented features are generated to support advanced analytics and machine learning.

### Features Created

#### Logistics Features

* Shipping Lead Time
* Delay Status
* Route Efficiency Score

#### Geographic Features

* Factory Mapping
* Factory Coordinates
* State Information
* Regional Classification

#### Time Features

* Order Month
* Order Quarter
* Seasonal Indicators

#### Business Metrics

* Profit Margin %
* Shipment Volume Metrics
* Route Performance Indicators

### Output

```text
featured_nassau_candy.csv
```

---

# 📊 Phase 3 — Exploratory Data Analysis

Comprehensive EDA is performed to understand data patterns and uncover operational insights.

### Analysis Performed

#### Data Quality Analysis

* Missing values
* Outlier detection
* Distribution analysis

#### Sales Analysis

* Revenue distribution
* Profit analysis
* Product performance

#### Shipping Analysis

* Lead time distribution
* Delay patterns
* Efficiency evaluation

#### Geographic Analysis

* State-wise performance
* Regional performance
* Route effectiveness

### Deliverables

* High-quality visualizations
* Summary tables
* Business insights
* Executive-level observations

---

# 🚚 Phase 4 — Ship Mode Performance Analysis

This phase evaluates shipping performance across delivery methods.

### Ship Modes Analyzed

* Same Day
* First Class
* Second Class
* Standard Class

### Key Metrics

* Average lead time
* Delay percentage
* Shipment volume
* Revenue contribution
* Profitability
* Route efficiency

### Business Outcome

Identify the most efficient and cost-effective shipping strategy.

---

# 🗺 Phase 5 — State-Level Analysis

State-wise shipping performance analysis is performed to identify geographic bottlenecks.

### Analysis Includes

* Top-performing states
* Poor-performing states
* Delay hotspots
* Revenue contribution by state
* Route efficiency rankings
* State-level KPI evaluation

### Business Outcome

Identify states requiring logistics optimization.

---

# 🌎 Phase 6 — Region-Level Analysis

Regional shipping performance is evaluated to understand broader geographic trends.

### Analysis Includes

* Regional lead times
* Delay comparison
* Revenue performance
* Profitability assessment
* Route efficiency comparison

### Business Outcome

Identify underperforming regions and strategic improvement opportunities.

---

# 🏭 Phase 7 — Factory-Level Analysis

Factory performance is analyzed to evaluate operational effectiveness.

### Key Metrics

* Shipment volume
* Revenue generated
* Profit contribution
* Delay frequency
* Route efficiency
* Regional coverage

### Business Outcome

Identify best-performing and underperforming factories.

---

# 📈 Phase 8 — Interactive Streamlit Dashboard

An interactive business intelligence dashboard is developed using Streamlit.

### Dashboard Features

#### Executive Summary

* High-level KPIs
* Operational overview

#### Ship Mode Analytics

* Delivery performance
* Delay trends

#### Geographic Analytics

* State analysis
* Region analysis

#### Factory Analytics

* Factory comparison
* Efficiency metrics

#### Interactive Filters

* Region filter
* State filter
* Factory filter
* Ship mode filter

### Run Dashboard

```bash
streamlit run 08_streamlit_dashboard.py
```

---

# 🤖 Phase 9 — Machine Learning

Predictive models are developed to forecast shipping performance.

## Classification Problem

### Target

```text
Delay Status
```

Classes:

* On Time
* Moderate Delay
* Delayed

### Models Used

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Gradient Boosting Classifier

---

## Regression Problem

### Target

```text
Shipping Lead Time
```

### Models Used

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor

---

## Evaluation Metrics

### Classification Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC AUC
* Confusion Matrix

### Regression Metrics

* MAE
* RMSE
* R² Score
* MAPE

---

# 🛠 Technologies Used

## Programming Language

* Python

## Data Analysis

* Pandas
* NumPy

## Visualization

* Matplotlib
* Seaborn
* Plotly

## Dashboard Development

* Streamlit

## Machine Learning

* Scikit-Learn
* SciPy
* Joblib

---

# 🚀 How to Run the Project

### Clone Repository

```bash
git clone https://https://github.com/satyam-ssingh/Factory-to-Customer-Shipping-Route-Efficiency-Analysis
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Project Pipeline

```bash
python 01_data_cleaning.py

python 02_feature_engineering.py

python 03_exploratory_data_analysis.py

python 04_ship_mode_analysis.py

python 05_state_analysis.py

python 06_region_analysis.py

python 07_factory_analysis.py

python 09_ml_training_and_evaluation.py
```

### Launch Dashboard

```bash
streamlit run 08_streamlit_dashboard.py
```

---

# 📌 Key Outcomes

* End-to-End Data Analytics Pipeline
* Business Intelligence Reporting
* Logistics Performance Evaluation
* Route Efficiency Analysis
* Geographic Bottleneck Detection
* Factory Performance Benchmarking
* Interactive Dashboard Development
* Predictive Machine Learning Models
* Actionable Business Recommendations

---

# 👨‍💻 Author

**Satyam Kumar Singh**

BCA Student | Data Analytics | Machine Learning | Business Intelligence

Passionate about solving real-world business problems using data analytics, visualization, and machine learning.

---

⭐ If you found this project useful, consider giving the repository a star.
