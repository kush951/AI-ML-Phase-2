
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix

students = pd.read_csv("data/students.csv")
jobs = pd.read_csv("data/jobs.csv")

dataset = []

for _, s in students.iterrows():
    s_skills = set(s["skills"].split(","))

    for _, j in jobs.iterrows():
        j_skills = set(j["required_skills"].split(","))

        overlap = len(s_skills.intersection(j_skills))
        total = len(j_skills)

        match_score = overlap / total

        label = 1 if (
            match_score >= 0.5 and
            s["verified_score"] >= j["min_score"]
        ) else 0

        dataset.append({
            "skill_overlap": match_score,
            "verified_score": s["verified_score"],
            "experience": s["experience"],
            "label": label
        })

df = pd.DataFrame(dataset)

X = df[["skill_overlap", "verified_score", "experience"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)

tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
fpr = fp / (fp + tn)

print("Precision:", precision)
print("Recall:", recall)
print("False Positive Rate:", round(fpr, 2))

joblib.dump(model, "model.pkl")

print("Model saved as model.pkl")
