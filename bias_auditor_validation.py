# TAKEN/ADAPTED FROM FROM BASHA ET AL'S WORK/CODE: https://osf.io/uydae/overview?view_only=20966206d38840ee907d576bb17fd4bd
import json
import re
from difflib import SequenceMatcher

# =============================
# CONFIGURATION
# =============================
INPUT_FILE = "llm_outputs_manually_final_gemini1.csv"
SAMPLE_FILE = "random_100_dataset_.csv"
OUTPUT_FILE = "random_100_evaluation_results_.csv"
SUMMARY_FILE = "random_100_summary_.csv"

SAMPLE_SIZE = 100
RANDOM_SEED = 42

MODEL_COLUMNS = ["gemini"]

termos_dict = [
    "lonely", "superficial", "imaginary", "inept", "lazy",
    "sick", "upset", "angry", "disgusting", "filthy",
    "nasty", "repulsive", "ugly", "shitty", "superb",
    "dark", "mawkish", "aground", "ironical", "freakish",
    "vile", "idiotic", "stupid", "dumb", "propitious",
    "sporty", "extraordinary", "contemptuous", "erroneous",
    "depressed", "jealous", "tired", "poor", "fat", "rotten",
    "horrible", "terrible", "awful", "moronic", "fascinating",
    "heartfelt", "dauntless", "confident", "stylish", "snappy",
    "better", "worse", "best", "worst"
]


# =============================
# SAMPLE 100 RANDOM ROWS
# =============================
df = pd.read_csv(INPUT_FILE)

if len(df) < SAMPLE_SIZE:
    raise ValueError(
        f"Input contains only {len(df)} rows; "
        f"cannot sample {SAMPLE_SIZE}."
    )

sample_df = df.sample(
    n=SAMPLE_SIZE,
    replace=False,
    random_state=RANDOM_SEED
)

sample_df.to_csv(SAMPLE_FILE, index=False)
print(f"Saved random sample to {SAMPLE_FILE}")


# =============================
# JSON LOADING
# =============================
def safe_load(x):
    if not isinstance(x, str) or not x.strip():
        return []

    try:
        data = json.loads(x)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except (json.JSONDecodeError, TypeError):
        try:
            data = json.loads(f"[{x}]")
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    return []


# =============================
# NORMALIZATION
# =============================
def normalize_text(x):
    if x is None:
        return ""

    x = str(x).lower()
    x = x.replace("”", '"').replace("“", '"')
    x = re.sub(r"\s+", "", x)
    return x.strip()


def normalize_type(t):
    if not t:
        return ""

    t = str(t).lower()
    t = t.replace("documentation/naming", "naming/documentation")
    t = t.replace("/documentation", "")
    t = t.replace("documentation/", "")

    return t.strip()


def normalize_stereotype_terms(text):
    if not text:
        return ""

    text = text.lower()
    text = text.replace("the adjective/characteristic", "")
    text = text.replace("group/attribute", "")

    return text


def clean_reasoning(text):
    if not text:
        return ""

    text = str(text).lower()
    text = text.replace("”", '"').replace("“", '"')

    # Remove term dictionary tuple annotations.
    text = re.sub(
        r"\(\s*'[^']+'\s*,\s*'(positive|negative|neutral|indefinite)'\s*\)",
        "",
        text
    )

    for term in termos_dict:
        if f"'{term}'" in text:
            return f"link '{term}'"

    return normalize_text(text)


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


# =============================
# MATCHING
# =============================
def match_biases(manual_list, model_list):

    used = set()

    type_scores = []
    code_scores = []
    reason_scores = []

    type_matches = 0

    for manual in manual_list:
        if not isinstance(manual, dict):
            continue

        best_j = None
        best_score = -1
        best_type = 0
        best_code = 0
        best_reason = 0

        m_type = normalize_type(manual.get("bias_type"))
        m_code = normalize_text(manual.get("biased_code"))

        m_reason = normalize_stereotype_terms(
            clean_reasoning(
                manual.get("impact_reasoning") or manual.get("issue")
            )
        )

        for j, predicted in enumerate(model_list):

            if j in used or not isinstance(predicted, dict):
                continue

            g_type = normalize_type(predicted.get("bias_type"))
            g_code = normalize_text(predicted.get("biased_code"))

            g_reason = normalize_stereotype_terms(
                clean_reasoning(
                    predicted.get("impact_reasoning")
                    or predicted.get("issue")
                )
            )

            type_match = int(
                m_type == g_type
                or m_type in g_type
                or g_type in m_type
            ) if m_type and g_type else 0

            code_sim = similarity(m_code, g_code)
            reason_sim = similarity(m_reason, g_reason)

            # Same matching strategy as the original script.
            score = (code_sim + reason_sim) / 2

            if score > best_score:
                best_score = score
                best_j = j
                best_type = type_match
                best_code = code_sim
                best_reason = reason_sim

        if best_j is not None:
            used.add(best_j)

            type_scores.append(best_type)
            code_scores.append(best_code)
            reason_scores.append(best_reason)

            type_matches += best_type

    return type_matches, type_scores, code_scores, reason_scores



# =============================
# EVALUATION
# =============================
results = []

summary_counts = {
    model: {
        "category": {"TP": 0, "FP": 0, "FN": 0, "TN": 0},
        "localization": {"TP": 0, "FP": 0, "FN": 0}
    }
    for model in MODEL_COLUMNS
}

for i, row in sample_df.iterrows():

    row_result = {"row": i}

    manual_list = [
        x for x in safe_load(row.get("justification", ""))
        if isinstance(x, dict) and x.get("bias_type")
    ]

    for model in MODEL_COLUMNS:

        model_list = [
            x for x in safe_load(row.get(f"{model}_parsed", ""))
            if isinstance(x, dict) and x.get("bias_type")
        ]

        # One-to-one snippet matching
        type_matches, type_s, code_s, reason_s = match_biases(
            manual_list, model_list
        )

        # -----------------------------
        # CATEGORY DETECTION
        # -----------------------------
        manual_set = {
            normalize_type(x.get("bias_type"))
            for x in manual_list
        }

        model_set = {
            normalize_type(x.get("bias_type"))
            for x in model_list
        }

        category_tp = len(manual_set & model_set)
        category_fp = len(model_set - manual_set)
        category_fn = len(manual_set - model_set)

        # -----------------------------
        # CODE LOCALIZATION
        # -----------------------------
        # Each matched prediction is a TP.
        # Unmatched predictions are FP.
        # Unmatched ground-truth snippets are FN.
        localization_tp = len(code_s)
        localization_fp = max(0, len(model_list) - localization_tp)
        localization_fn = max(0, len(manual_list) - localization_tp)

        # Accumulate counts across all sampled rows
        cat = summary_counts[model]["category"]
        cat["TP"] += category_tp
        cat["FP"] += category_fp
        cat["FN"] += category_fn

        loc = summary_counts[model]["localization"]
        loc["TP"] += localization_tp
        loc["FP"] += localization_fp
        loc["FN"] += localization_fn

        # Per-row similarity metrics
        row_result[f"{model}_avg_code_similarity"] = (
            sum(code_s) / len(code_s) if code_s else 0
        )

        row_result[f"{model}_avg_reason_similarity"] = (
            sum(reason_s) / len(reason_s) if reason_s else 0
        )

        row_result[f"{model}_avg_type_match"] = (
            sum(type_s) / len(type_s) if type_s else 0
        )

    results.append(row_result)


# =============================
# SAVE PER-ROW RESULTS
# =============================
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved per-row results to {OUTPUT_FILE}")


# =============================
# OVERALL SUMMARY
# =============================
summary = []

for model in MODEL_COLUMNS:

    for task in ["category", "localization"]:

        counts = summary_counts[model][task]

        tp = counts["TP"]
        fp = counts["FP"]
        fn = counts["FN"]

        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0
        )

        # Micro-averaged Jaccard accuracy
        accuracy = (
            tp / (tp + fp + fn)
            if (tp + fp + fn) else 0
        )

        summary.append({
            "Model": model,
            "Evaluation_Task": task,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Accuracy": accuracy
        })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(SUMMARY_FILE, index=False)

print("\nOVERALL RESULTS:")
print(summary_df.to_string(index=False))

print(f"\nSaved summary to {SUMMARY_FILE}")