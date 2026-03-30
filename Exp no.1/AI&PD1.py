
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load Dataset
df = pd.read_csv("student_depression_datasets.csv")

# 2. Drop unnecessary column
df = df.drop("id", axis=1)

# 3. Handle categorical data
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# 4. Split features and target
X = df.drop("Depression", axis=1)
y = df["Depression"]

# 5. Train-test split (UNSEEN DATA)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 7. Predict on unseen data
y_pred = model.predict(X_test)

# 8. Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# 9. Test with new unseen input
sample = X_test.iloc[0:1]
prediction = model.predict(sample)

print("Sample Prediction:", prediction)
