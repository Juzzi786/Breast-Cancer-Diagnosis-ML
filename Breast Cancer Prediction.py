import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.inspection import permutation_importance

# Load dataset
df = pd.read_csv(r"C:\Users\juzer\OneDrive\Desktop\python\Breast Cancer\Breast_Cancer.csv")
df.columns = df.columns.str.strip()  # Remove spaces around column names
print(df.columns.tolist()) 

# Copy and preprocess data
data = df.copy()
label_encoders = {}
for col in data.columns:
    if data[col].dtype == 'object':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

# Manually specify all features except target
all_features = ['Age', 'Race', 'Marital Status', 'T Stage', 'N Stage', '6th Stage', 'differentiate', 'Grade', 'A Stage', 'Tumor Size', 'Estrogen Status', 'Progesterone Status', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months']

X = data[all_features]
y = data['Status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier()
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    results[name] = {"Train Accuracy": train_acc, "Test Accuracy": test_acc}
    
    print(f"\n{name}")
    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print("Classification Report:\n", classification_report(y_test, y_test_pred))
    
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Voting ensemble
voting_clf = VotingClassifier(estimators=[
    ('lr', models["Logistic Regression"]),
    ('rf', models["Random Forest"]),
    ('xgb', models["XGBoost"]),
    ('svm', models["SVM"]),
    ('knn', models["KNN"]),
], voting='soft')

voting_clf.fit(X_train_scaled, y_train)

y_train_pred_ensemble = voting_clf.predict(X_train_scaled)
y_test_pred_ensemble = voting_clf.predict(X_test_scaled)

ensemble_train_acc = accuracy_score(y_train, y_train_pred_ensemble)
ensemble_test_acc = accuracy_score(y_test, y_test_pred_ensemble)

results["Voting Ensemble"] = {
    "Train Accuracy": ensemble_train_acc,
    "Test Accuracy": ensemble_test_acc
}

print("\nVoting Ensemble (Hybrid)")
print(f"Train Accuracy: {ensemble_train_acc * 100:.2f}%")
print(f"Test Accuracy: {ensemble_test_acc * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_test_pred_ensemble))

cm_ensemble = confusion_matrix(y_test, y_test_pred_ensemble)
sns.heatmap(cm_ensemble, annot=True, fmt="d", cmap="Greens")
plt.title("Voting Ensemble - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --- New Part: Permutation Feature Importance on Voting Ensemble ---
print("\nCalculating Permutation Feature Importance for Voting Ensemble...")

perm_importance = permutation_importance(
    voting_clf, X_test_scaled, y_test, 
    n_repeats=10, random_state=42, scoring='accuracy'
)

# Sort features by importance
sorted_idx = perm_importance.importances_mean.argsort()[::-1]

plt.figure(figsize=(10,6))
sns.barplot(x=perm_importance.importances_mean[sorted_idx], y=[all_features[i] for i in sorted_idx])
plt.title("Permutation Feature Importance - Voting Ensemble")
plt.xlabel("Mean decrease in accuracy")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Summary Table
print("\nSummary of Model Accuracies:")
summary_df = pd.DataFrame(results).T
summary_df = summary_df.applymap(lambda x: f"{x*100:.2f}%")
print(summary_df)
