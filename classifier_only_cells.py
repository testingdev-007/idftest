# Classifier-only Python in Excel cells: TF-IDF + logistic regression, fixed seed 42
# Paste each block into its own Python cell, one per row, top to bottom.
# Needs one Excel Table named Complaints (complaint_id, text, category). No rules table required.

# ================= CELL 1: Load the data =================
import pandas as pd

# Excel Table named Complaints, with columns: complaint_id, text, category
df = xl("Complaints[#All]", headers=True)
df["category"].value_counts()

# ================= CELL 2: Fixed-seed train/test split =================
from sklearn.model_selection import train_test_split

SEED = 42  # the fixed seed: change it and the split changes

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["category"]
)
pd.DataFrame({"rows": [len(train_df), len(test_df)]}, index=["train", "test"])

# ================= CELL 3: Train TF-IDF + logistic regression =================
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODEL_VERSION = "tfidf-lr-v1"

def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                                  min_df=2, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, C=10, random_state=SEED)),
    ])

model = build_model()
model.fit(train_df["text"], train_df["category"])
"trained: " + MODEL_VERSION

# ================= CELL 4: Check accuracy on held-out complaints =================
from sklearn.metrics import accuracy_score, classification_report

pred = model.predict(test_df["text"])
print_acc = round(accuracy_score(test_df["category"], pred), 3)

report = pd.DataFrame(
    classification_report(test_df["category"], pred, output_dict=True, zero_division=0)
).T.round(2)
report.loc["OVERALL ACCURACY", "precision"] = print_acc
report

# ================= CELL 5: Prove it is reproducible (5 retrains) =================
import hashlib

def fingerprint(m):
    labels = m.predict(df["text"])
    probs = m.predict_proba(df["text"]).round(6)
    return hashlib.md5(("|".join(labels) + str(probs.tolist())).encode()).hexdigest()[:12]

runs = []
for i in range(5):
    m = build_model()
    m.fit(train_df["text"], train_df["category"])
    runs.append({"run": i + 1, "fingerprint": fingerprint(m)})

out = pd.DataFrame(runs)
out["identical to run 1"] = out["fingerprint"] == out["fingerprint"].iloc[0]
out

# ================= CELL 6: Classify new complaints (low confidence goes to a person) =================
THRESHOLD = 0.60   # below this confidence, a person decides

new = pd.DataFrame({"text": [
    "I can't get into the app and the reset email never comes",
    "Someone I don't know took money out of my account",
    "You charged me a fee nobody told me about",
    "I have waited six weeks and nothing has happened",
]})

probs = model.predict_proba(new["text"])
top = probs.argmax(axis=1)
new["label"] = [model.classes_[i] if probs[n, i] >= THRESHOLD else "Needs review"
                for n, i in enumerate(top)]
new["confidence"] = probs.max(axis=1).round(2)
new["model_version"] = MODEL_VERSION
new
