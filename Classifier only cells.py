# Classifier-only Python in Excel cells: TF-IDF + logistic regression, fixed seed 42
# Paste each block into its own Python cell, one per row, top to bottom.
# Needs one Excel Table named Complaints (complaint_id, text, category). No rules table required.
# Cells 7 and 9 each return all 900 rows: keep the space below and to the right of them empty.
# Cells 8 and 9 contain random results on purpose: your numbers change every run.
# Cells 12 and 13 need a second Excel Table named Runs (complaint_id, run_1, run_2).

# ================= CELL 1: Load the data =================
import pandas as pd

# Excel Table named Complaints, with columns: complaint_id, text, category
df = xl("Complaints[#All]", headers=True)
df["category"].value_counts()

# ================= CELL 2: Fixed-seed train/test split (720 train, 180 test) =================
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

# NOTE: with scikit-learn's default solver (lbfgs), LogisticRegression has no random step,
# so random_state below changes nothing. It is kept as insurance in case you switch solver.
# The seed that really matters is the one in train_test_split (cell 2).
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

# ================= CELL 7: Final output: all 900 rows =================
# FINAL OUTPUT: all 900 complaints (model only, no rules table needed)
import numpy as np

THRESHOLD = 0.60
probs = model.predict_proba(df["text"])

final = df[["complaint_id", "text"]].copy()
final["true_label"]       = df["category"]
final["model_label"]      = model.classes_[probs.argmax(axis=1)]
final["model_confidence"] = probs.max(axis=1).round(3)
final["final_label"]      = np.where(final["model_confidence"] >= THRESHOLD, final["model_label"], "Needs review")
final["used_for"]         = np.where(final["complaint_id"].isin(test_df["complaint_id"]), "test", "train")
# "Needs review" counts as not correct, so this is stricter than the model-only accuracy in cell 4
final["correct"]          = final["final_label"] == final["true_label"]
final["model_version"]    = MODEL_VERSION
final

# ================= CELL 8: The seed experiment: what changes if a seed is left out? =================
# THE SEED EXPERIMENT: what changes if a seed is left out?
from sklearn.metrics import accuracy_score

def train_and_label(split_seed, clf_seed):
    tr, te = train_test_split(df, test_size=0.2, random_state=split_seed, stratify=df["category"])
    m = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, C=10, random_state=clf_seed)),
    ])
    m.fit(tr["text"], tr["category"])
    labels = m.predict(df["text"])                                   # label all 900 complaints
    acc = accuracy_score(te["category"], m.predict(te["text"]))      # accuracy on ITS test set
    return labels, fingerprint(m), acc

baseline_labels, _, _ = train_and_label(42, 42)

experiments = [
    ("A. Seed 42 everywhere",               42,   42),
    ("B. No seed on the train/test split",  None, 42),
    ("C. No seed on the classifier only",   42,   None),
    ("D. No seeds at all",                  None, None),
]
rows = []
for name, split_seed, clf_seed in experiments:
    for run in range(1, 6):
        labels, fp, acc = train_and_label(split_seed, clf_seed)
        rows.append({"experiment": name, "run": run, "fingerprint": fp,
                     "test accuracy": round(acc, 3),
                     "complaints labelled differently from A (of 900)": int((labels != baseline_labels).sum())})
pd.DataFrame(rows)

# ================= CELL 9: Seedless variant: all 900 rows with no seed on the split =================
# SEEDLESS VARIANT of cell 7: same model, same 900 complaints, but NO seed on the train/test split.
# Re-run this cell and the labels change. That is what a missing seed does.
tr, te = train_test_split(df, test_size=0.2, stratify=df["category"])      # <-- no random_state

m = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000, C=10)),                        # <-- no random_state
])
m.fit(tr["text"], tr["category"])

probs_s = m.predict_proba(df["text"])
te_ids = set(te["complaint_id"])

final_seedless = df[["complaint_id", "text"]].copy()
final_seedless["true_label"]                = df["category"]
final_seedless["seeded_final_label"]        = final["final_label"]           # from cell 7 (seed 42)
final_seedless["seedless_model_label"]      = m.classes_[probs_s.argmax(axis=1)]
final_seedless["seedless_model_confidence"] = probs_s.max(axis=1).round(3)
final_seedless["seedless_final_label"]      = np.where(final_seedless["seedless_model_confidence"] >= THRESHOLD,
                                                       final_seedless["seedless_model_label"], "Needs review")
final_seedless["used_for_this_run"]         = np.where(final_seedless["complaint_id"].isin(te_ids), "test", "train")
final_seedless["seedless_correct"]          = final_seedless["seedless_final_label"] == final_seedless["true_label"]
final_seedless["changed_vs_seeded"]         = final_seedless["seedless_final_label"] != final_seedless["seeded_final_label"]
final_seedless

# ================= CELL 10: Compare the two 900-row outputs =================
# COMPARE THE TWO 900-ROW OUTPUTS: seeded (cell 7) against seedless (cell 9)
cmp = pd.DataFrame({
    "complaint_id":   final["complaint_id"],
    "true_label":     final["true_label"],
    "confidence":     final["model_confidence"],         # the seeded model's confidence
    "seeded_label":   final["final_label"],
    "seedless_label": final_seedless["seedless_final_label"],
})
cmp["same"] = cmp["seeded_label"] == cmp["seedless_label"]
cmp["band"] = pd.cut(cmp["confidence"], bins=[0, 0.6, 0.8, 0.9, 1.01], right=False,
                     labels=["under 60%", "60-80%", "80-90%", "90%+"])

def summarise(view, group, d):
    return {"view": view, "group": group, "complaints": len(d),
            "same label": int(d["same"].sum()), "different label": int((~d["same"]).sum()),
            "% same": round(100 * d["same"].mean(), 1) if len(d) else None}

rows = [summarise("Overall", "All 900 complaints", cmp)]
for band in cmp["band"].cat.categories:
    rows.append(summarise("By seeded confidence", band, cmp[cmp["band"] == band]))
for cat in sorted(cmp["true_label"].unique()):
    rows.append(summarise("By true category", cat, cmp[cmp["true_label"] == cat]))
pd.DataFrame(rows)

# ================= CELL 11: Run log: the audit record to keep beside frozen labels =================
# RUN LOG: the audit record to keep beside your frozen labels
import sklearn, datetime

pd.DataFrame([{
    "run_date":      datetime.date.today().isoformat(),
    "model_version": MODEL_VERSION,
    "split_seed":    SEED,
    "rows":          len(df),
    "train_rows":    len(train_df),
    "test_rows":     len(test_df),
    "fingerprint":   fingerprint(model),
    "scikit-learn":  sklearn.__version__,
    "pandas":        pd.__version__,
}]).T.rename(columns={0: "value"})

# ================= CELL 12: Compare any two runs (needs an Excel Table named Runs) =================
# COMPARE ANY TWO RUNS. Needs an Excel Table named Runs with three columns:
#   complaint_id, run_1, run_2
# Use it for: a frozen column against a fresh recalculation, or Copilot's labels from two separate runs.
# Capitalisation and stray spaces are ignored; only the label itself is compared.
run_pair = xl("Runs[#All]", headers=True)
# Only complaints with a label in BOTH columns are compared (blank cells must not count as agreement)
run_pair = run_pair.dropna(subset=["run_1", "run_2"])
run_pair = run_pair[(run_pair["run_1"].astype(str).str.strip() != "") & (run_pair["run_2"].astype(str).str.strip() != "")].copy()
run_pair["same"] = run_pair["run_1"].astype(str).str.strip().str.lower() == run_pair["run_2"].astype(str).str.strip().str.lower()

pd.DataFrame({
    "measure": ["complaints compared", "same label both times", "different label", "agreement %"],
    "value":   [str(len(run_pair)), str(int(run_pair["same"].sum())), str(int((~run_pair["same"]).sum())),
                f"{100 * run_pair['same'].mean():.1f}%"],
})

# ================= CELL 13: Which complaints differ? =================
# WHICH COMPLAINTS DIFFER? (uses the Runs table from cell 14)
differ = run_pair[~run_pair["same"]].merge(df[["complaint_id", "text", "category"]], on="complaint_id", how="left")
differ[["complaint_id", "text", "category", "run_1", "run_2"]].rename(columns={"category": "true_label"})
