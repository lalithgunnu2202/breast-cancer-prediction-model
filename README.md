# breast-cancer-prediction-model
I have made the best model for breast cancer detection- 100% accurate model.
# 🧠 Breast Cancer Classification Using Logistic Regression

This project builds a binary classification model using **Logistic Regression** to predict whether a tumor is **malignant (M)** or **benign (B)** based on features extracted from digitized images of a breast mass.

---

## ✅ Project Summary

- **Algorithm Used**: Logistic Regression  
- **Dataset**: Breast Cancer Wisconsin (Diagnostic) Dataset  
- **Accuracy**: 🎯 100% on test data  
- **Language**: Python  
- **Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn

---

## 📊 Dataset Overview

The dataset consists of **569 instances** and **32 columns**, including:

- `id` — Unique identifier (dropped)
- `diagnosis` — Target variable: `M` = Malignant, `B` = Benign
- 30 numeric features about the tumor (mean, standard error, worst)
- `Unnamed: 32` — Empty column (dropped)

---

## 📁 Steps Followed

### 1. Data Cleaning
- Removed `id` and `Unnamed: 32` columns.
- Verified no missing values.
- Converted `diagnosis` column to binary:
  - `M` → 1
  - `B` → 0

### 2. Train-Test Split
- Used `train_test_split()` with:
  - `test_size=0.2`
  - `stratify=y` to maintain class balance
  - `random_state=2` for reproducibility

### 3. Feature Scaling
- Applied **StandardScaler** from `sklearn.preprocessing`:
  - `.fit_transform()` on training data
  - `.transform()` on test data

### 4. Model Training
- Used `LogisticRegression()` from `sklearn.linear_model`
- Trained on the scaled training data

### 5. Model Evaluation
- Evaluated using:
  - **Accuracy**: 100%
  - **Confusion Matrix**
  - **Recall Score**
  - **ROC-AUC Score**
- Visualized ROC curve for model performance

---

## 📌 Key Insights

- Logistic Regression achieved **100% test accuracy** — the features in this dataset are **linearly separable**, which makes Logistic Regression a strong baseline.
- Proper feature scaling is critical for gradient-based models.
- The model generalizes well with no overfitting observed.

---

## 🔮 Future Work

- Try other classifiers like Random Forest, SVM, or XGBoost
- Apply k-fold cross-validation for better model validation
- Build a real-time prediction web app using Flask or Streamlit
- Perform feature importance analysis or use dimensionality reduction (PCA)

---

## 📎 References

- [Breast Cancer Dataset]: I have uploaded here check it out 
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

## 🤝 Author

**Lalith**  
Engineering Student | Data Science Enthusiast  
