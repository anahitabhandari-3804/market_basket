This version addresses the **class imbalance issue** using **SMOTE (Synthetic Minority Over-sampling Technique)** and **adjusts class weights in XGBoost** to improve predictions for both classes. 🚀

---

### **🔹 Key Improvements**

✅ **Fixed Class Imbalance** using **SMOTE**

✅ **Added Class Weight Adjustment** in **XGBoost**

✅ **Improved Data Cleaning** (Removed refunds, handled missing values)

✅ **Evaluated Model with F1-score, Precision, and Recall**

✅ **Displayed Balanced Class Distribution**

classification report 

![image](https://github.com/user-attachments/assets/6ee58648-3c62-490b-b986-7369c50cf221)


### **🔹 What Each Metric Means**

✅ **Precision (`0.81` for class `0`, `1.00` for class `1`)**

- When the model predicts **low purchase behavior (`0`)**, it's correct **81%** of the time.
- When it predicts **high purchase behavior (`1`)**, it's **always correct (`100%`).**

✅ **Recall (`1.00` for class `0`, `0.77` for class `1`)**

- **Class `0`:** The model correctly identifies **all low-purchase cases (100%)**.
- **Class `1`:** It captures **77% of actual high-purchase cases** (so, **23% of high-purchase customers are misclassified as low-purchase**).

✅ **F1-Score (`0.90` for class `0`, `0.87` for class `1`)**

- **Balanced score between precision and recall.**
- **Good performance overall!**

✅ **Accuracy (`0.88` or `88%`)**

- **Overall, the model makes the correct prediction 88% of the time.**

output :

![image](https://github.com/user-attachments/assets/6a04a760-db1d-40af-a082-7d70ebc7f3a9)


 **For example Rule 1:**

- **If a customer buys:** `"HAND WARMER UNION JACK"`
- **Then they might also buy:** `"HAND WARMER OWL DESIGN"`
- **Support:** `0.0238` (2.38% of transactions had "HAND WARMER UNION JACK")
- **Jaccard:** `0.2599` (Measure of similarity)
- **Certainty Factor:** `0.457` (Confidence level in the rule)
