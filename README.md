# 🧬 Cancer Risk Prediction with XGBoost (Semi-Supervised)

This project builds a cancer risk prediction model using a **semi-supervised machine learning** pipeline. It combines powerful feature selection and ensemble learning with unlabeled data leveraging, to improve performance with limited supervision.

---

## 📁 Project ipynb Files

| File | Description |
|------|-------------|
| `preprocessor.ipynb` | 📌 Preprocessing: cleaning, SMOTETomek balancing, and feature selection using RFE |
| `model_training.ipynb` | 🧠 Model training: XGBoost + hyperparameter tuning + pseudo-labeling |

---

## 📦 Requirements

All packages used are listed in `requirements.txt`.

> ✅ Tip: You can run this project directly in [Google Colab](https://colab.research.google.com/) without installing anything locally!

---

## 📊 Dataset

The dataset is from the **[ML-AI Hackathon 2025 on Kaggle](https://www.kaggle.com/competitions/ml-ai-hackathon-2025/data)**.
Final test predictions saved as `result_on_selected_test.csv` and is uploaded.
 

## 🧩 Challenges Faced

- 🔍 **Class Imbalance**  
  We tackled severe class imbalance using `SMOTETomek`, which combines oversampling and cleaning.

- 🧪 **Limited Labeled Data**  
  With only 280 labeled samples and 250 unlabeled ones, we used **pseudo-labeling** to build a semi-supervised learning pipeline.

- ⚙️ **Feature Selection**  
  With over **14,500 features**, selecting the most informative 500–1000 was non-trivial. We used **Recursive Feature Elimination (RFE)** with Random Forest and Logistic Regression.

- 🧮 **Time-Consuming Tuning**  
  Hyperparameter tuning with **GridSearchCV** (5-fold cross-validation × 216 combinations) was computationally heavy.

- 🧭 **Missing Ground Truth on Test Set**  
  Since the test set had no labels, we relied on confidence scores, ROC-AUC plots, and validation performance to evaluate our model.

---

## 🏁 Results Summary

- ✅ **Used**: `XGBoostClassifier` with `multi:softprob`
- 🏗️ Final model trained using combined real + pseudo-labeled data
- 📁 Final test predictions saved as `result_on_selected_test.csv`

---

## 🤝 Acknowledgments

Thanks to **Kaggle**, **XGBoost**, and the IITG ML-AI Hackathon organizers.

---

> 🔗 For any questions, feel free to raise an issue or drop a message.


