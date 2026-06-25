# DevelopersHub AI/ML Engineering Internship

## Task 3: Heart Disease Prediction

### Objective

The objective of this project is to construct a machine learning classification pipeline to predict whether a patient is at risk of heart disease based on their physiological and clinical health profile. Utilizing the famous **UCI Cleveland Heart Disease dataset**, we clean medical records, handle missing entries, map multi-class disease diagnostics into a binary classification problem, and benchmark a linear model (**Logistic Regression**) against a non-linear model (**Decision Tree Classifier**).

Additionally, this project analyzes feature importances to expose the key diagnostic indicators that contribute most heavily to cardiovascular risk.

### Dataset Overview

* **Data Source:** UCI Machine Learning Repository (Cleveland Dataset)
* **Sample Count:** 303 patient records (pre-cleaning)
* **Clinical Features ($X$):** The dataset comprises 13 distinct physiological indicators:
   1. `age`: Age in years
   2. `sex`: Sex (1 = male; 0 = female)
   3. `cp`: Chest pain type (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 4 = asymptomatic)
   4. `trestbps`: Resting blood pressure (in mm Hg on admission to the hospital)
   5. `chol`: Serum cholesterol in mg/dl
   6. `fbs`: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
   7. `restecg`: Resting electrocardiographic results (0 = normal; 1 = ST-T wave abnormality; 2 = left ventricular hypertrophy)
   8. `thalach`: Maximum heart rate achieved
   9. `exang`: Exercise-induced angina (1 = yes; 0 = no)
  10. `oldpeak`: ST depression induced by exercise relative to rest
  11. `slope`: The slope of peak exercise ST segment (1 = upsloping; 2 = flat; 3 = downsloping)
  12. `ca`: Number of major vessels (0-3) colored by fluoroscopy
  13. `thal`: Thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
* **Target Variable ($y$):** Diagnostic binary status:
  * `0`: Healthy / No Disease
  * `1`: Heart Disease Present (mapped from levels 1-4)

---

### Installation & Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/numanabubakar/AI-ML-Internship-Task.git
cd AI-ML-Internship-Task/Task 3
   ```

2. **Install required dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

3. **Run the classification pipeline:**
   ```bash
   python task3_heart_disease.py
   ```

---

### Methodology & Modeling Architecture

#### 1. Clinical Data Ingestion & Sanitization
* **Missing Value Recovery:** Missing values labeled as `?` are dropped to preserve data integrity.
* **Target Simplification:** Maps original multi-class disease severity (0-4) to a simple binary diagnosis target $y \in \{0, 1\}$.

#### 2. Stratified Partitioning & Standard Scaling
* **Stratified Split:** Guarantees balance is maintained across both training ($80%$) and validation ($20%$) environments.
* **Feature Standardization:** Computes $z$-score transformations to scale numerical attributes to standard variance intervals.

---

### Key Skills Demonstrated
* Clinical diagnostic preprocessing
* Dual machine learning classifier benchmarking (Linear vs. Tree Models)
* Advanced clinical predictive modeling diagnostic metric analysis (ROC-AUC Curves, Confusion Matrices)