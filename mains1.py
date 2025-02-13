import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load dataset
file_path = r"C:\Users\DELL\Desktop\xyz\market basket\OnlineRetail.csv"
df = pd.read_csv(file_path, encoding="ISO-8859-1")

# Select relevant columns and clean data
df = df[['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']]
df.dropna(subset=['CustomerID'], inplace=True)  # Remove rows with missing CustomerID
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]  # Remove negative values (Refunds)
df['InvoiceNo'] = df['InvoiceNo'].astype(str)
df['StockCode'] = df['StockCode'].astype(str)

# Create basket for Market Basket Analysis
basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

# Apply FP-Growth Algorithm
frequent_itemsets = fpgrowth(basket, min_support=0.01, use_colnames=True)

# Generate Association Rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
print("Top 5 Association Rules:\n", rules.head())

# Prepare Data for Machine Learning
X = basket.values
y = df.groupby("InvoiceNo")["Quantity"].sum().apply(lambda x: 1 if x > 5 else 0).values  # Target: High Purchase Volume

# Print class distribution before balancing
unique, counts = np.unique(y, return_counts=True)
print("\nBefore Balancing - Class Distribution:", dict(zip(unique, counts)))

# Handle Class Imbalance using SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Print class distribution after balancing
unique, counts = np.unique(y_resampled, return_counts=True)
print("\nAfter Balancing - Class Distribution:", dict(zip(unique, counts)))

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Calculate scale_pos_weight for XGBoost (to handle imbalance)
scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

# Train XGBoost Model with Class Weights
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, scale_pos_weight=scale_pos_weight)
xgb_model.fit(X_train, y_train)

# Predictions & Evaluation
ml_predictions = xgb_model.predict(X_test)

print("\n🔹 **XGBoost Model Evaluation:**")
print("Accuracy:", accuracy_score(y_test, ml_predictions))
print("\nClassification Report:\n", classification_report(y_test, ml_predictions))
