
# 💳 Credit Card Fraud Detection using Machine Learning

This project aims to build and evaluate machine learning models to detect fraudulent credit card transactions using historical transaction data. Since fraud transactions are very rare (~0.17%), the dataset is highly imbalanced, making the detection task challenging and realistic.

---

## 📌 Objective

To build a classification model that can:
- Accurately identify fraudulent transactions (Class = 1)
- Minimize false negatives (i.e., actual fraud predicted as not fraud)
- Perform well even with imbalanced data (99.8% non-fraud)

---

## 📂 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Shape**: 284,807 rows × 31 columns
- **Features**:
  - `V1` to `V28`: PCA-transformed features
  - `Time`: Time elapsed since the first transaction
  - `Amount`: Transaction amount
  - `Class`: Target (0 = Not Fraud, 1 = Fraud)

---

## ⚙️ Tools & Libraries Used

- Python
- pandas, NumPy, seaborn, matplotlib
- scikit-learn (Logistic Regression, Naive Bayes, Decision Tree, Random Forest)
- StandardScaler
- Confusion Matrix, Classification Report

---

## 🧪 ML Models Applied

| Model                   | Accuracy | Strength                                                                 |
|------------------------|----------|--------------------------------------------------------------------------|
| Logistic Regression     | ~99.9%   | High accuracy, but poor fraud detection due to imbalance                |
| Naive Bayes             | Best     | Lower false negatives, simple and fast, best fraud detection            |
| Decision Tree Classifier| Good     | Performs better than logistic regression                                |
| Random Forest Classifier| Very Good| Reduces false negatives significantly, robust and interpretable         |

---

## 📊 Model Evaluation

- **Accuracy is not enough!** Due to class imbalance, we rely on:
  - **Confusion Matrix**
  - **Recall** (to catch more fraud)
  - **Precision** (to avoid false alarms)
  - **F1-Score** (balance between recall and precision)

Example Confusion Matrix for Naive Bayes:
```
[[71040    38]
 [   20    65]]
```

- Only 20 actual frauds were missed (compared to 44 in Logistic Regression)

---

## ✅ Conclusion

- **Naive Bayes** performed the best in this case, especially in reducing false negatives.
- **Random Forest** and **Decision Tree** also outperformed **Logistic Regression**.
- Logistic Regression showed high accuracy but failed to detect many frauds.
- In real-world fraud detection, minimizing **missed frauds (false negatives)** is critical.

---

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/ybthummar/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. Install the required libraries (if needed):
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Run the notebook:
- Open `Practical_1.ipynb` in Jupyter Notebook or Google Colab.
- Make sure `creditcard.csv` is in the correct path.

---

## 💡 Future Improvements

- Apply **SMOTE** or **undersampling** to improve model balance.
- Try **XGBoost**, **LightGBM**, or **deep learning models**.
- Add **SHAP** or **LIME** for explainability.
- Deploy the model as an API.

---

## 📬 Contact

Created by [Yug Thummar](mailto:yugthummar001@gmail.com)  
GitHub: [ybthummar](https://github.com/ybthummar)
# CreditCardFraudDetection
# CreditCardFraudDetection
