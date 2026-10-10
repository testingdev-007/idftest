"""
Train once, freeze to a file, classify with the frozen file.
Run:  python train_classifier_local.py
Needs: pip install pandas scikit-learn joblib
"""
import json, hashlib
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

SEED = 42
MODEL_VERSION = "tfidf-lr-v1"
MODEL_FILE = f"complaint_classifier_{MODEL_VERSION}.joblib"

df = pd.read_csv("sample_complaints.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["category"])

# NOTE: with the default solver (lbfgs), LogisticRegression has no random step, so its
# random_state changes nothing. The seed that really matters is the one in train_test_split.
model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000, C=10, random_state=SEED)),
])
model.fit(train_df["text"], train_df["category"])

acc = accuracy_score(test_df["category"], model.predict(test_df["text"]))
print(f"Test accuracy: {acc:.3f}")

# FREEZE: save the trained model. From now on you load this file; you never retrain by accident.
joblib.dump(model, MODEL_FILE)
# Two different fingerprints, for two different jobs:
#  - file hash: proves THIS saved file has not been swapped or edited (a retrain writes a new file)
#  - prediction fingerprint: proves two models behave identically on the same data
file_hash = hashlib.sha256(open(MODEL_FILE, "rb").read()).hexdigest()[:16]
blob = "|".join(model.predict(df["text"])) + str(model.predict_proba(df["text"]).round(6).tolist())
pred_fp = hashlib.md5(blob.encode()).hexdigest()[:12]
json.dump({"version": MODEL_VERSION, "seed": SEED, "test_accuracy": round(acc, 3),
           "file": MODEL_FILE, "file_sha256_prefix": file_hash,
           "prediction_fingerprint": pred_fp},
          open(f"{MODEL_FILE}.meta.json", "w"), indent=2)
print("Saved", MODEL_FILE, "| file hash:", file_hash, "| prediction fingerprint:", pred_fp)

# USE: load the frozen file and label new complaints (with a confidence threshold)
frozen = joblib.load(MODEL_FILE)
new = pd.Series(["I can't get into the app and the reset email never comes",
                 "Someone I don't know took money out of my account"])
probs = frozen.predict_proba(new)
for text, p in zip(new, probs):
    top = p.argmax()
    label = frozen.classes_[top] if p[top] >= 0.60 else "Needs review"
    print(f"{label:22s} conf={p[top]:.2f}  <- {text}")
