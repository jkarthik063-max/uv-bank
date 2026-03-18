
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Universal Bank - Personal Loan Prediction Dashboard")

df = pd.read_csv("UniversalBank.csv")

st.subheader("Dataset Overview")
st.write(df.head())

X = df.drop(columns=["Personal Loan", "ID", "ZIPCode"])
y = df["Personal Loan"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = []

st.subheader("Model Performance")

plt.figure()

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    report = classification_report(y_test, y_pred, output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    results.append([
        name,
        report["accuracy"],
        report["1"]["precision"],
        report["1"]["recall"],
        report["1"]["f1-score"]
    ])

plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
st.pyplot(plt)

results_df = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1"])
st.write(results_df)

st.subheader("Confusion Matrix (Random Forest)")
rf = models["Random Forest"]
y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", ax=ax)
st.pyplot(fig)

st.subheader("Upload Test Data for Prediction")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    test_df = pd.read_csv(uploaded_file)
    preds = rf.predict(test_df.drop(columns=["ID","ZIPCode"]))
    test_df["Predicted Personal Loan"] = preds
    st.write(test_df.head())

    csv = test_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
