
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import joblib

data = pd.DataFrame({
    "skill_overlap": [0.9,0.8,0.3,0.2,0.7,0.1],
    "experience_gap": [0,1,4,5,1,6],
    "label": [1,1,0,0,1,0]
})

X = data[["skill_overlap","experience_gap"]]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

models = {
    "logistic": LogisticRegression(),
    "random_forest": RandomForestClassifier(),
    "svm": SVC()
}

best_model = None
best_f1 = 0

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    f1 = f1_score(y_test, pred)

    print(name, f1)

    if f1 > best_f1:
        best_f1 = f1
        best_model = model

joblib.dump(best_model, "backend/saved_models/best_model.pkl")

print("Best model saved.")
