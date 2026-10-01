import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/emails.csv")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.25, random_state=42, stratify=df["label"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

samples = [
    "Congratulations, you have won a free cash prize!",
    "Please send me the project report before lunch."
]

print("\nSample predictions:")
for text in samples:
    print(f"{text} -> {model.predict([text])[0]}")
