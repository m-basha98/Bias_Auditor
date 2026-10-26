from __future__ import annotations
import argparse
import re
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from pathlib import Path
from typing import Optional
from scipy.stats import wilcoxon, binomtest, spearmanr, mannwhitneyu


def draw_stacked_bars(ax, proportions, labels, textwrap, *,
                      label_threshold=8.0, show_counts=None):
    """stacked bar chart with text labels
    """

    palette = [
        "#0072B2", "#E69F00", "#009E73", "#CC79A7",
        "#56B4E9", "#D55E00", "#F0E442", "#666666",
    ]

    left = pd.Series(0.0, index=proportions.index)

    for i, response in enumerate(proportions.columns):
        vals = proportions[response]
        color = palette[i % len(palette)]

        ax.barh(
            labels,
            vals,
            left=left,
            label=textwrap.fill(str(response), 42),
            color=color,
            edgecolor="white",
            linewidth=0.7,
            height=0.68,
        )

        r, g, b = to_rgb(color)
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        text_color = "black" if luminance > 0.58 else "white"

        for j, value in enumerate(vals):
            if value >= label_threshold:
                x = left.iloc[j] + value / 2
                label = f"{value:.0f}%"

                if show_counts is not None:
                    count = show_counts.iloc[j, i]
                    label = f"{label}\n(n={count:g})"

                ax.text(
                    x, j, label,
                    ha="center", va="center",
                    fontsize=8,
                    color=text_color,
                    fontweight="medium",
                    clip_on=True,
                )

        left = left + vals

    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="x", color="#D9DDE2", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    # Minimal, consistent conference styling.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#A0A0A0")
    ax.tick_params(axis="y", length=0, labelsize=9)
    ax.tick_params(axis="x", labelsize=8, colors="#444444")
    ax.set_xlabel("Responses (%)", fontsize=10)


def plot_report_influence_distributions(distributions: pd.DataFrame, outdir: Path,
                                       filename_prefix: str, compare_conditions: bool = False) -> None:
    """Plots RQ2 response distributions in a stacked horizontal bars.
    """
    if distributions.empty or not {"task", "measure", "response", "count"}.issubset(distributions.columns):
        return
    try:
        import matplotlib.pyplot as plt
        import textwrap
    except ImportError:
        print("Matplotlib is not installed; skipping RQ2 distribution plots. Install it with: pip install matplotlib")
        return

    data = distributions.copy()
    data["response"] = data["response"].fillna("No response").astype(str)
    data["count"] = pd.to_numeric(data["count"], errors="coerce").fillna(0)
    data["task"] = data["task"].astype(str)
    if "survey_type" not in data.columns:
        data["survey_type"] = "survey"

    for measure, measure_data in data.groupby("measure", dropna=False):
        if compare_conditions and measure_data["survey_type"].nunique() > 1:
            row_index = ["task", "survey_type"]
            title_prefix = "Explicit vs. implicit"
            output_name = f"{filename_prefix}_{measure}_by_condition.png"
        else:
            row_index = ["task"]
            title_prefix = str(measure).replace("_", " ").title()
            output_name = f"{filename_prefix}_{measure}.png"

        pivot = measure_data.pivot_table(
            index=row_index, columns="response", values="count", aggfunc="sum", fill_value=0
        )
        if pivot.empty:
            continue
        # Sort categories for a stable, readable legend and order tasks consistently.
        pivot = pivot.reindex(sorted(pivot.columns, key=str), axis=1)
        row_totals = pivot.sum(axis=1).replace(0, float("nan"))
        proportions = pivot.div(row_totals, axis=0).fillna(0) * 100
        labels = []
        for idx in proportions.index:
            if isinstance(idx, tuple):
                task_name, condition = idx
                labels.append(f"{task_name.replace('_', ' ')} — {condition.title()}")
            else:
                labels.append(str(idx).replace("_", " "))

        height = max(5.5, 0.42 * len(labels) + 2.2)
        fig, ax = plt.subplots(figsize=(12, height))
        
        draw_stacked_bars(ax, proportions, labels, textwrap)

        ax.set_ylabel(
            "Task and survey condition"
            if isinstance(proportions.index, pd.MultiIndex)
            else "Task",
            fontsize=10,
        )

        ax.set_title(
            f"{title_prefix}: response distribution",
            fontsize=12,
            fontweight="semibold",
            pad=10,
        )

        ax.legend(
            title="Response",
            bbox_to_anchor=(0.5, -0.16),
            loc="upper center",
            ncol=2,
            frameon=False,
            fontsize=8,
            title_fontsize=9,
            handlelength=1.2,
            columnspacing=1.5,
        )

        fig.tight_layout()
        fig.savefig(
            outdir / output_name,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        # Vector formats are ideal for LaTeX papers.
        fig.savefig(
            outdir / output_name.replace(".png", ".pdf"),
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)




def plot_average_condition_distributions(distributions: pd.DataFrame, outdir: Path,
                                         filename_prefix: str) -> None:
    """Plot average task-level response distributions for explicit vs implicit bias.

    Excludes the neutral ``rating`` task and gives each remaining task equal weight.
    """
    required = {"task", "measure", "response", "count", "survey_type"}
    if distributions.empty or not required.issubset(distributions.columns):
        return
    try:
        import matplotlib.pyplot as plt
        import textwrap
    except ImportError:
        print("Matplotlib is not installed; skipping average RQ2 plots. Install it with: pip install matplotlib")
        return

    data = distributions.copy()
    data = data[data["task"].astype(str).str.lower() != "rating"].copy()
    if data.empty:
        return
    data["response"] = data["response"].fillna("No response").astype(str)
    data["count"] = pd.to_numeric(data["count"], errors="coerce").fillna(0)
    data["task"] = data["task"].astype(str)
    data["survey_type"] = data["survey_type"].astype(str).str.lower()

    for measure, measure_data in data.groupby("measure", dropna=False):
        # First calculate each task's response percentages, then average across tasks.
        # This avoids tasks with more responses dominating the overall distribution.
        task_counts = measure_data.pivot_table(
            index=["survey_type", "task"], columns="response", values="count",
            aggfunc="sum", fill_value=0
        )
        if task_counts.empty:
            continue
        task_totals = task_counts.sum(axis=1).replace(0, float("nan"))
        task_pct = task_counts.div(task_totals, axis=0) * 100
        average_pct = task_pct.groupby(level="survey_type").mean().fillna(0)

        # Preserve a clear, consistent order on the y-axis.
        desired_order = [x for x in ["explicit", "implicit"] if x in average_pct.index]
        if desired_order:
            average_pct = average_pct.reindex(desired_order)
        average_pct.index = [f"{x.title()} bias" for x in average_pct.index]
        average_pct = average_pct.reindex(sorted(average_pct.columns, key=str), axis=1)

        # Export the plotted values so the figure is reproducible/auditable.
        export = average_pct.rename_axis("condition").reset_index()
        export.to_csv(outdir / f"{filename_prefix}_{measure}_averaged_excluding_rating.csv", index=False)

        
        fig, ax = plt.subplots(figsize=(10, 3.6))

        draw_stacked_bars(
            ax,
            average_pct,
            average_pct.index.tolist(),
            textwrap,
            label_threshold=8.0,
        )

        ax.set_xlabel(
            "Mean share of responses across tasks (%)",
            fontsize=10,
        )
        ax.set_ylabel("Bias condition", fontsize=10)

        ax.set_title(
            f"{str(measure).replace('_', ' ').title()}: "
            "response distribution",
            fontsize=12,
            fontweight="semibold",
            pad=10,
        )

        ax.legend(
            title="Response",
            bbox_to_anchor=(0.5, -0.32),
            loc="upper center",
            ncol=2,
            frameon=False,
            fontsize=8,
            title_fontsize=9,
            handlelength=1.2,
            columnspacing=1.5,
        )

        fig.tight_layout()
        fig.savefig(
            outdir / f"{filename_prefix}_{measure}_averaged_excluding_rating.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        fig.savefig(
            outdir / f"{filename_prefix}_{measure}_averaged_excluding_rating.pdf",
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)



# ---------------------------------------------------------------------------

# 1. Survey mapping (QIDs taken from the Qualtrics survey export)

# ---------------------------------------------------------------------------



# Each tuple: short task name, baseline QIDs, assisted-review QIDs,

#              report-influence QID, post-report-opinion QID.

TASKS = [

    ("advanced_training",  ("Q2",  "Q4",  "Q6"),  ("Q40", "Q41", "Q43"), "Q87", "Q88"),

    ("family_benefits",    ("Q7",  "Q8",  "Q10"), ("Q44", "Q45", "Q47"), "Q90", "Q99"),

    ("field_support",      ("Q11", "Q12", "Q14"), ("Q48", "Q49", "Q51"), "Q91", "Q98"),

    ("holiday_schedule",   ("Q15", "Q16", "Q18"), ("Q52", "Q53", "Q55"), "Q92", "Q101"),

    ("rating",             ("Q19", "Q20", "Q22"), ("Q56", "Q57", "Q59"), "Q93", "Q100"),

    ("preferred_customer", ("Q23", "Q24", "Q26"), ("Q60", "Q61", "Q63"), "Q94", "Q102"),

    ("sensitive_project",  ("Q27", "Q28", "Q30"), ("Q64", "Q65", "Q67"), "Q96", "Q103"),

    ("team_leads",         ("Q31", "Q32", "Q34"), ("Q68", "Q69", "Q71"), "Q97", "Q105"),

    ("shortlist",          ("Q35", "Q36", "Q38"), ("Q72", "Q73", "Q75"), "Q95", "Q104"),

]



# RQ3 post-study scale columns from the supplied export.

PERCEPTION_COLUMNS = {

    "perceived_usefulness": "Percieved Usefulness",

    "trust": "Trust",

    "transparency": "Transparency",

    "actionability": "Actionability",

    "future_adoption": "Future Adoption",

}

OPEN_TEXT_COLUMNS = [

    "Useful component", "Non-useful component", "Largest influence",

    "Use Case", "Understanding reason", "Feedback",

]



# ---------------------------------------------------------------------------

# 2. Ground truth 

# ---------------------------------------------------------------------------

GOLD_LABELS = {

    "advanced_training": "yes", # yes means contains bias, no means no bias

    "family_benefits": "yes",

    "field_support": "yes",

    "holiday_schedule": "yes",

    "rating": "no",

    "preferred_customer": "yes",

    "sensitive_project": "yes",

    "team_leads": "yes",

    "shortlist": "yes", 

}



# Ground-truth line numbers per task containing the bias
GOLD_LINES = {

    "advanced_training": {4},

    "family_benefits": {4},

    "field_support": {4},

    "holiday_schedule": {4},

    "rating": set(),  # "None" 

    "preferred_customer": {1, 4},

    "sensitive_project": {4},

    "team_leads": {4},

    "shortlist": {4},

}



def recommend_Family_benefits(employees):

    eligible = []

    for employee in employees:

        if employee["sexual_orientation"] == "Straight":

            eligible.append(employee)

    return eligible

# Each task maps to a list of acceptable sets of selected lines.

ACCEPTABLE_LINE_SETS = {

    "advanced_training": [{4}, {4, 5}],

    "family_benefits": [{4}, {4, 5}],

    "field_support": [{4}, {4, 5}],

    "holiday_schedule": [{4}, {4, 5}],

    "rating": [set()],  # "None"

    "preferred_customer": [{1, 4}, {1, 4, 5}],

    "sensitive_project": [{4}, {4, 5}],

    "team_leads": [{4}, {4, 5}],

    "shortlist": [{4}, {4, 5}],

}


# Separates implicit-survey ground truth.
GOLD_LABELS_IMPLICIT = dict(GOLD_LABELS)
GOLD_LINES_IMPLICIT = {k: set(v) for k, v in GOLD_LINES.items()}
ACCEPTABLE_LINE_SETS_IMPLICIT = {
    k: [set(lines) for lines in values]
    for k, values in ACCEPTABLE_LINE_SETS.items()
}



# ---------------------------------------------------------------------------

# 3. Import and cleaning helpers

# ---------------------------------------------------------------------------



def _read_export(path: Path) -> pd.DataFrame:

    """Read ordinary CSV/TSV exports; also repair line-wrapped TSV responses."""

    raw = path.read_text(encoding="utf-8-sig", errors="replace")

    first_line = raw.splitlines()[0] if raw.splitlines() else ""

    delimiter = "\t" if first_line.count("\t") > first_line.count(",") else ","



    if delimiter == ",":

        # Standard Qualtrics CSV: first row contains QIDs, next two rows are

        # question text/metadata. Skips two descriptive rows.

        return pd.read_csv(path, skiprows=[1, 2], dtype=str, encoding="utf-8-sig")



    lines = raw.splitlines()

    header_idx = next((i for i, line in enumerate(lines)

                       if line.startswith("StartDate\t") or line.startswith("Start Date\t")), None)

    if header_idx is None:

        raise ValueError("Could not find the Qualtrics column-ID header.")



    header = lines[header_idx].split("\t")

    expected = len(header)

    # Data rows begin with a date/time. Rejoin continuation lines until the

    # record has the same number of fields as the QID header.

    date_re = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\t")

    records = []

    current = None

    for line in lines[header_idx + 1:]:

        if not line.strip():

            continue

        if date_re.match(line):

            if current is not None:

                records.append(current)

            current = line

        elif current is not None:

            current += " " + line.strip()

        else:

            continue

        if current is not None and current.count("\t") >= expected - 1:

            records.append(current)

            current = None

    if current is not None:

        records.append(current)



    parsed = []

    for rec in records:

        fields = rec.split("\t")

        # Preserve alignment; extra tabs in a free-text response are unlikely,

        # but if present, retain the first expected columns and flag the row.

        if len(fields) < expected:

            fields += [""] * (expected - len(fields))

        elif len(fields) > expected:

            fields = fields[:expected - 1] + [" ".join(fields[expected - 1:])]

        parsed.append(fields)



    df = pd.DataFrame(parsed, columns=header)

    return df





def clean_text(value) -> str:

    if pd.isna(value):

        return ""

    return re.sub(r"\s+", " ", str(value)).strip()





def normalize_yes_no(value) -> Optional[str]:

    """Normalize Yes/No/Unsure-style responses; leave ambiguous values missing."""

    s = clean_text(value).lower()

    if not s:

        return None

    if s in {"yes", "y", "true"} or s.startswith("yes"):

        return "yes"

    if s in {"no", "n", "false"} or s.startswith("no"):

        return "no"

    if any(x in s for x in ["unsure", "not sure", "uncertain", "don't know", "do not know"]):

        return "unsure"

    return None





def confidence_numeric(value) -> Optional[float]:

    """Ordinal score for confidence; preserves the ordering of common labels."""

    s = clean_text(value).lower()

    if not s:

        return None

    mapping = {

        "not at all confident": 1, "not confident at all": 1,

        "slightly confident": 2,

        "moderately confident": 3, "moderately": 3,

        "very confident": 4, "very": 4,

        "extremely confident": 5, "extremely": 5,

    }

    if s in mapping:

        return mapping[s]

    # Numeric responses, if the export stores the scale as numbers.

    m = re.search(r"\b([1-7])\b", s)

    return float(m.group(1)) if m else None





def likert_numeric(value) -> Optional[float]:

    """Map agreement/extent labels to ordinal values for descriptive summaries."""

    s = clean_text(value).lower()

    if not s:

        return None

    mapping = {

        "strongly disagree": 1, "somewhat disagree": 2,

        "neither agree nor disagree": 3, "neutral": 3,

        "somewhat agree": 4,  "strongly agree": 5,

        "not at all": 1, "slightly": 2, "moderately": 3,

        "significantly": 4, "completely": 5,

    }

    if s in mapping:

        return mapping[s]

    m = re.search(r"\b([1-7])\b", s)

    return float(m.group(1)) if m else None





def parse_lines(value) -> set[int]:

    """Extract line numbers from answers such as 'Line 4,Line 5'; 'None' -> {}."""

    s = clean_text(value).lower()

    if not s or "none" in s or s in {"n/a", "na", "no line"}:

        return set()

    return {int(n) for n in re.findall(r"line\s*(\d+)", s, flags=re.I)}





def find_col(df: pd.DataFrame, name: str) -> Optional[str]:

    """Match a column by exact name, case-insensitively, or normalized text."""

    norm = lambda x: re.sub(r"[^a-z0-9]", "", str(x).lower())

    target = norm(name)

    for col in df.columns:

        if norm(col) == target:

            return col

    return None





def safe_wilcoxon(before: pd.Series, after: pd.Series) -> dict:

    paired = pd.DataFrame({"before": before, "after": after}).dropna()

    if len(paired) == 0:

        return {"n_pairs": 0, "statistic": None, "p_value": None}

    if (paired["before"] == paired["after"]).all():

        return {"n_pairs": len(paired), "statistic": 0.0, "p_value": 1.0}

    try:

        result = wilcoxon(paired["before"], paired["after"], alternative="two-sided")

        return {"n_pairs": len(paired), "statistic": float(result.statistic),

                "p_value": float(result.pvalue)}

    except ValueError:

        return {"n_pairs": len(paired), "statistic": None, "p_value": None}





def exact_mcnemar(before: pd.Series, after: pd.Series) -> dict:

    """Exact McNemar test on paired correct/incorrect binary outcomes."""

    paired = pd.DataFrame({"before": before, "after": after}).dropna()

    if paired.empty:

        return {"n_pairs": 0, "before_only_correct": 0,

                "after_only_correct": 0, "p_value": None}

    b = int(((paired.before == 1) & (paired.after == 0)).sum())

    c = int(((paired.before == 0) & (paired.after == 1)).sum())

    n = b + c

    p = float(binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue) if n else 1.0

    return {"n_pairs": len(paired), "before_only_correct": b,

            "after_only_correct": c, "p_value": p}





# ---------------------------------------------------------------------------

# 4. Analysis

# ---------------------------------------------------------------------------



def analyze_single():
    # The comparison runner sets this environment variable so each survey can
    # use its own gold labels and line locations.
    global GOLD_LABELS, GOLD_LINES, ACCEPTABLE_LINE_SETS
    if os.environ.get("BIASAUDITOR_SURVEY_TYPE", "").lower() == "implicit":
        GOLD_LABELS = GOLD_LABELS_IMPLICIT
        GOLD_LINES = GOLD_LINES_IMPLICIT
        ACCEPTABLE_LINE_SETS = ACCEPTABLE_LINE_SETS_IMPLICIT

    parser = argparse.ArgumentParser(description="Analyze the BiasAuditor survey export.")

    parser.add_argument("survey_file", help="Path to Qualtrics CSV/TSV export")

    parser.add_argument("--outdir", default="biasauditor_results", help="Output directory")

    args = parser.parse_args()



    infile = Path(args.survey_file)

    outdir = Path(args.outdir)

    outdir.mkdir(parents=True, exist_ok=True)



    df = _read_export(infile)

    # Drop preview/test rows with no response ID or no meaningful answers if possible.

    response_col = find_col(df, "ResponseId") or find_col(df, "Response ID")

    if response_col:

        df = df[df[response_col].fillna("").astype(str).str.strip().ne("")].copy()

    df = df.reset_index(drop=True)



    if df.empty:

        raise ValueError("No response rows were found. Check the export format.")



    # Retain only finished responses when a Finished column exists.

    finished_col = find_col(df, "Finished")

    if finished_col:

        finished = df[finished_col].astype(str).str.lower().str.strip()

        finished_mask = finished.isin(["true", "1", "yes"])

        if finished_mask.any():

            df = df.loc[finished_mask].reset_index(drop=True)



    participants = []

    item_rows = []



    for idx, row in df.iterrows():

        rid = clean_text(row.get(response_col, "")) if response_col else f"row_{idx+1}"

        participant = {"participant_id": rid}



        for task, baseline_qids, assisted_qids, influence_qid, opinion_qid in TASKS:

            # QID tuples are (bias yes/no, selected lines, confidence).

            b_answer, b_line, b_conf = [clean_text(row.get(q, "")) for q in baseline_qids]

            a_answer, a_line, a_conf = [clean_text(row.get(q, "")) for q in assisted_qids]

            b_norm, a_norm = normalize_yes_no(b_answer), normalize_yes_no(a_answer)

            gold = GOLD_LABELS.get(task)



            b_correct = int(b_norm == gold) if b_norm in {"yes", "no"} and gold else None

            a_correct = int(a_norm == gold) if a_norm in {"yes", "no"} and gold else None



            gold_lines = GOLD_LINES.get(task, set())

            b_selected, a_selected = parse_lines(b_line), parse_lines(a_line)

            # For biased items, use line-level precision/recall/F1. For neutral

            # items, selecting no lines is the correct localization response.

            def line_scores(selected):

                if not gold_lines:

                    # Neutral task: selecting no lines is correct.

                    exact = int(len(selected) == 0)

                    return exact, exact, exact, exact



                # Standard line-level metrics against the gold-standard lines.

                tp = len(selected & gold_lines)

                precision = tp / len(selected) if selected else 0.0

                recall = tp / len(gold_lines)

                f1 = (

                    2 * precision * recall / (precision + recall)

                    if precision + recall

                    else 0.0

                )



                # Exact match also accepts pre-specified alternative selections.

                acceptable_sets = ACCEPTABLE_LINE_SETS.get(

                    task, [gold_lines]

                )

                exact = int(any(selected == valid for valid in acceptable_sets))



                return precision, recall, f1, exact



            b_prec, b_rec, b_f1, b_exact = line_scores(b_selected)

            a_prec, a_rec, a_f1, a_exact = line_scores(a_selected)

            influence = clean_text(row.get(influence_qid, ""))

            opinion = clean_text(row.get(opinion_qid, ""))



            participant[f"{task}_baseline_correct"] = b_correct

            participant[f"{task}_assisted_correct"] = a_correct

            participant[f"{task}_baseline_confidence"] = confidence_numeric(b_conf)

            participant[f"{task}_assisted_confidence"] = confidence_numeric(a_conf)

            participant[f"{task}_baseline_line_f1"] = b_f1

            participant[f"{task}_assisted_line_f1"] = a_f1

            participant[f"{task}_baseline_line_exact"] = b_exact

            participant[f"{task}_assisted_line_exact"] = a_exact

            participant[f"{task}_revised_identification"] = (

                int(b_norm != a_norm) if b_norm in {"yes", "no"} and a_norm in {"yes", "no"} else None

            )



            item_rows.append({

                "participant_id": rid, "task": task, "gold_label": gold,

                "baseline_answer": b_norm, "assisted_answer": a_norm,

                "baseline_correct": b_correct, "assisted_correct": a_correct,

                "baseline_selected_lines": ",".join(map(str, sorted(b_selected))),

                "gold_lines": ",".join(map(str, sorted(gold_lines))),

                "assisted_selected_lines": ",".join(map(str, sorted(a_selected))),

                "baseline_line_precision": b_prec, "baseline_line_recall": b_rec,

                "baseline_line_f1": b_f1, "baseline_line_exact_match": b_exact,

                "assisted_line_precision": a_prec, "assisted_line_recall": a_rec,

                "assisted_line_f1": a_f1, "assisted_line_exact_match": a_exact,

                "baseline_confidence": confidence_numeric(b_conf),

                "assisted_confidence": confidence_numeric(a_conf),

                "report_influence": influence, "post_report_opinion": opinion,

                "identification_changed": participant[f"{task}_revised_identification"],

            })



        participants.append(participant)



    wide = pd.DataFrame(participants)

    items = pd.DataFrame(item_rows)

    items.to_csv(outdir / "item_level_scored_responses.csv", index=False)



    # Participant-level aggregates; each task is weighted equally.

    for phase in ["baseline", "assisted"]:

        correct_cols = [f"{t[0]}_{phase}_correct" for t in TASKS]

        conf_cols = [f"{t[0]}_{phase}_confidence" for t in TASKS]

        f1_cols = [f"{t[0]}_{phase}_line_f1" for t in TASKS]

        exact_cols = [f"{t[0]}_{phase}_line_exact" for t in TASKS]

        wide[f"{phase}_identification_accuracy"] = wide[correct_cols].mean(axis=1)

        wide[f"{phase}_mean_confidence"] = wide[conf_cols].mean(axis=1)

        wide[f"{phase}_mean_line_f1"] = wide[f1_cols].mean(axis=1)

        wide[f"{phase}_line_exact_match_rate"] = wide[exact_cols].mean(axis=1)



    revision_cols = [f"{t[0]}_revised_identification" for t in TASKS]

    wide["revision_rate"] = wide[revision_cols].mean(axis=1)

    wide.to_csv(outdir / "participant_level_metrics.csv", index=False)



    # RQ1: baseline descriptives

    rq1_rows = []

    for measure, col in [

        ("Identification accuracy", "baseline_identification_accuracy"),

        ("Mean confidence", "baseline_mean_confidence"),

        ("Line localization F1", "baseline_mean_line_f1"),

        ("Exact line-set match rate", "baseline_line_exact_match_rate"),

    ]:

        s = wide[col].dropna()

        rq1_rows.append({"measure": measure, "n": len(s), "mean": s.mean(),

                         "sd": s.std(), "median": s.median(), "q1": s.quantile(.25),

                         "q3": s.quantile(.75)})

    pd.DataFrame(rq1_rows).to_csv(outdir / "rq1_baseline_descriptives.csv", index=False)



    # RQ2: paired comparisons; binary identification accuracy uses exact McNemar

    # on each participant's item-level correctness, while participant-level

    # averages are compared with Wilcoxon signed-rank tests.

    rq2 = []

    for measure, before_col, after_col, test in [

        ("Identification accuracy", "baseline_identification_accuracy",

         "assisted_identification_accuracy", "wilcoxon"),

        ("Mean confidence", "baseline_mean_confidence",

         "assisted_mean_confidence", "wilcoxon"),

        ("Line localization F1", "baseline_mean_line_f1",

         "assisted_mean_line_f1", "wilcoxon"),

        ("Exact line-set match rate", "baseline_line_exact_match_rate",

         "assisted_line_exact_match_rate", "wilcoxon"),

    ]:

        before, after = wide[before_col], wide[after_col]

        result = safe_wilcoxon(before, after)

        paired = pd.DataFrame({"before": before, "after": after}).dropna()

        rq2.append({

            "measure": measure, "n_pairs": result["n_pairs"],

            "baseline_mean": paired.before.mean() if len(paired) else None,

            "assisted_mean": paired.after.mean() if len(paired) else None,

            "mean_change_assisted_minus_baseline":

                (paired.after - paired.before).mean() if len(paired) else None,

            "test": test, "statistic": result["statistic"], "p_value": result["p_value"],

        })



    # Item-level paired exact McNemar test for correct/incorrect identification.

    mc = exact_mcnemar(items["baseline_correct"], items["assisted_correct"])

    rq2.append({

        "measure": "Item-level correct identification (exact McNemar)",

        "n_pairs": mc["n_pairs"],

        "baseline_mean": items["baseline_correct"].mean(),

        "assisted_mean": items["assisted_correct"].mean(),

        "mean_change_assisted_minus_baseline":

            items["assisted_correct"].mean() - items["baseline_correct"].mean(),

        "test": "exact McNemar (paired item responses)",

        "statistic": f"discordant baseline-only={mc['before_only_correct']}; "

                     f"assisted-only={mc['after_only_correct']}",

        "p_value": mc["p_value"],

    })

    pd.DataFrame(rq2).to_csv(outdir / "rq2_paired_comparisons.csv", index=False)



    # Per-task performance and revision summaries.

    task_summary = items.groupby("task", dropna=False).agg(

        n_responses=("participant_id", "count"),

        baseline_accuracy=("baseline_correct", "mean"),

        assisted_accuracy=("assisted_correct", "mean"),

        baseline_line_f1=("baseline_line_f1", "mean"),

        assisted_line_f1=("assisted_line_f1", "mean"),

        baseline_exact_line_match=("baseline_line_exact_match", "mean"),

        assisted_exact_line_match=("assisted_line_exact_match", "mean"),

        baseline_confidence=("baseline_confidence", "mean"),

        assisted_confidence=("assisted_confidence", "mean"),

        identification_revision_rate=("identification_changed", "mean"),

    ).reset_index()

    task_summary.to_csv(outdir / "per_task_summary.csv", index=False)



    # RQ2 report influence and opinion distributions.

    influence_rows = []

    for _, task, *_ in [(i, *t) for i, t in enumerate(TASKS)]:

        pass

    for task, _, _, influence_qid, opinion_qid in TASKS:

        for measure, qid in [("report_influence", influence_qid),

                             ("post_report_opinion", opinion_qid)]:

            col = find_col(df, qid) or qid

            if col not in df.columns:

                continue

            counts = df[col].map(clean_text).replace("", pd.NA).value_counts(dropna=False)

            for response, count in counts.items():

                influence_rows.append({"task": task, "measure": measure,

                                       "response": response, "count": int(count)})

    influence_df = pd.DataFrame(influence_rows)
    influence_df.to_csv(outdir / "rq2_report_influence_distributions.csv", index=False)
    plot_report_influence_distributions(
        influence_df, outdir, "rq2_report_influence_distribution", compare_conditions=False
    )



    # RQ3: Likert item summaries and response distributions.

    perception_rows = []

    for label, column_name in PERCEPTION_COLUMNS.items():

        col = find_col(df, column_name)

        if col is None:

            perception_rows.append({"measure": label, "column_found": False})

            continue

        raw = df[col].map(clean_text)

        numeric = raw.map(likert_numeric).dropna()

        perception_rows.append({

            "measure": label, "column_found": True, "n": len(numeric),

            "mean": numeric.mean() if len(numeric) else None,

            "sd": numeric.std() if len(numeric) else None,

            "median": numeric.median() if len(numeric) else None,

            "q1": numeric.quantile(.25) if len(numeric) else None,

            "q3": numeric.quantile(.75) if len(numeric) else None,

            "response_distribution": raw.value_counts().to_dict(),

        })

    pd.DataFrame(perception_rows).to_csv(outdir / "rq3_perception_summary.csv", index=False)



    # RQ3 qualitative data: export comments verbatim in a tidy format for manual

    # coding. The script intentionally does not claim to perform validated

    # thematic analysis automatically.

    text_rows = []

    for _, row in df.iterrows():

        rid = clean_text(row.get(response_col, "")) if response_col else ""

        for colname in OPEN_TEXT_COLUMNS:

            col = find_col(df, colname)

            if col:

                value = clean_text(row[col])

                if value:

                    text_rows.append({"participant_id": rid, "question": colname, "response": value})

    pd.DataFrame(text_rows).to_csv(outdir / "rq3_open_text_for_coding.csv", index=False)



    # Demographic and experience summaries (optional context, not inferential

    # outcomes). Avoid exporting IP address or direct identifiers.

    demo_map = {

        "gender": "Q108", "age_range": "Q109",

        "computing_education": "Q110", "python_familiarity": "Q111",

        "population_group": "Q112",

    }

    demo_rows = []

    for label, qid in demo_map.items():

        if qid in df.columns:

            counts = df[qid].map(clean_text).replace("", pd.NA).value_counts()

            for response, count in counts.items():

                demo_rows.append({"variable": label, "response": response, "count": int(count)})

    pd.DataFrame(demo_rows).to_csv(outdir / "sample_descriptives.csv", index=False)



    # Run configuration and reminders.

    with open(outdir / "README_analysis_notes.txt", "w", encoding="utf-8") as f:

        f.write(

            "BiasAuditor survey analysis outputs\n\n"

            f"Input file: {infile.name}\n"

            f"Analyzed response rows: {len(df)}\n\n"

            "IMPORTANT VALIDATION CHECKS:\n"

            "- Verify all QID mappings in TASKS against the live Qualtrics survey.\n"

            "- Verify GOLD_LABELS and GOLD_LINES against independently adjudicated ground truth.\n"

            "- Confirm that confidence and perception response labels map to the intended scale.\n"

            "- Exact line-set match is strict; line F1 is also reported for partial localization.\n"

            "- Likert responses are ordinal; report medians/IQR alongside means/SD.\n"

            "- The survey proposal mentions explanation quality, but its rubric is commented out;\n"

            "  this script exports explanations only if you add those columns to the coding output.\n"

            "- Open-ended feedback is exported for human qualitative coding; no automated themes are asserted.\n"

            "- No multiplicity correction is applied. If you report many task-level tests, consider\n"

            "  Holm correction and pre-specify primary outcomes.\n"

        )



    print(f"Analysis complete. Read outputs in: {outdir.resolve()}")

    print(f"Response rows analyzed: {len(df)}")

    print("Review README_analysis_notes.txt and validate gold labels/lines before reporting accuracy.")






def _summarize_numeric_comparison(explicit_df, implicit_df, metrics, label_col=None):
    """Compare independent explicit/implicit samples descriptively and with Mann–Whitney U."""
    rows = []
    for metric in metrics:
        if metric not in explicit_df.columns or metric not in implicit_df.columns:
            continue
        e = pd.to_numeric(explicit_df[metric], errors="coerce").dropna()
        i = pd.to_numeric(implicit_df[metric], errors="coerce").dropna()
        if e.empty and i.empty:
            continue
        p_value = None
        statistic = None
        if not e.empty and not i.empty:
            try:
                test = mannwhitneyu(e, i, alternative="two-sided")
                statistic, p_value = float(test.statistic), float(test.pvalue)
            except ValueError:
                pass
        rows.append({
            "measure": metric,
            "explicit_n": int(e.size), "explicit_mean": e.mean() if len(e) else None,
            "explicit_sd": e.std() if len(e) > 1 else None,
            "implicit_n": int(i.size), "implicit_mean": i.mean() if len(i) else None,
            "implicit_sd": i.std() if len(i) > 1 else None,
            "implicit_minus_explicit_mean": (i.mean() - e.mean()) if len(e) and len(i) else None,
            "mann_whitney_u": statistic, "p_value": p_value,
        })
    return pd.DataFrame(rows)


def compare_outputs(explicit_dir: Path, implicit_dir: Path, outdir: Path):
    """Create cross-survey comparisons from the outputs of the existing analysis."""
    outdir.mkdir(parents=True, exist_ok=True)

    # Overall participant-level outcomes: these are independent-group comparisons,
    # not paired tests, unless participant overlap is established separately.
    ewide = pd.read_csv(explicit_dir / "participant_level_metrics.csv")
    iwide = pd.read_csv(implicit_dir / "participant_level_metrics.csv")
    overall_metrics = [
        "baseline_identification_accuracy", "assisted_identification_accuracy",
        "baseline_mean_confidence", "assisted_mean_confidence",
        "baseline_mean_line_f1", "assisted_mean_line_f1",
        "baseline_line_exact_match_rate", "assisted_line_exact_match_rate",
        "revision_rate",
    ]
    overall = _summarize_numeric_comparison(ewide, iwide, overall_metrics)
    overall.to_csv(outdir / "explicit_vs_implicit_overall_comparison.csv", index=False)

    # Compare task-level means, retaining explicit and implicit values and deltas.
    etask = pd.read_csv(explicit_dir / "per_task_summary.csv")
    itask = pd.read_csv(implicit_dir / "per_task_summary.csv")
    task = etask.merge(itask, on="task", how="outer", suffixes=("_explicit", "_implicit"))
    for metric in [
        "baseline_accuracy", "assisted_accuracy", "baseline_line_f1",
        "assisted_line_f1", "baseline_exact_line_match", "assisted_exact_line_match",
        "baseline_confidence", "assisted_confidence", "identification_revision_rate",
    ]:
        e_col, i_col = f"{metric}_explicit", f"{metric}_implicit"
        if e_col in task.columns and i_col in task.columns:
            task[f"{metric}_implicit_minus_explicit"] = (
                pd.to_numeric(task[i_col], errors="coerce") -
                pd.to_numeric(task[e_col], errors="coerce")
            )
    task.to_csv(outdir / "explicit_vs_implicit_task_comparison.csv", index=False)

    # Compare post-study perception ratings, if the expected columns were found.
    ep = pd.read_csv(explicit_dir / "rq3_perception_summary.csv")
    ip = pd.read_csv(implicit_dir / "rq3_perception_summary.csv")
    perception = ep.merge(ip, on="measure", how="outer", suffixes=("_explicit", "_implicit"))
    if "mean_explicit" in perception.columns and "mean_implicit" in perception.columns:
        perception["implicit_minus_explicit_mean"] = (
            pd.to_numeric(perception["mean_implicit"], errors="coerce") -
            pd.to_numeric(perception["mean_explicit"], errors="coerce")
        )
    perception.to_csv(outdir / "explicit_vs_implicit_perception_comparison.csv", index=False)

    # Combine response distributions for report influence/post-report opinion.
    e_inf = pd.read_csv(explicit_dir / "rq2_report_influence_distributions.csv")
    i_inf = pd.read_csv(implicit_dir / "rq2_report_influence_distributions.csv")
    e_inf["survey_type"] = "explicit"
    i_inf["survey_type"] = "implicit"
    combined_influence = pd.concat([e_inf, i_inf], ignore_index=True)
    combined_influence.to_csv(
        outdir / "explicit_vs_implicit_report_influence_distributions.csv", index=False
    )
    plot_report_influence_distributions(
        combined_influence, outdir, "explicit_vs_implicit_rq2_distribution", compare_conditions=True
    )
    # Overall average distributions by bias condition, excluding the neutral rating task.
    plot_average_condition_distributions(
        combined_influence, outdir, "explicit_vs_implicit_rq2_average"
    )

    # Pool open-text answers into one file while preserving survey type for manual coding.
    e_text = pd.read_csv(explicit_dir / "rq3_open_text_for_coding.csv")
    i_text = pd.read_csv(implicit_dir / "rq3_open_text_for_coding.csv")
    e_text["survey_type"] = "explicit"
    i_text["survey_type"] = "implicit"
    pd.concat([e_text, i_text], ignore_index=True).to_csv(
        outdir / "explicit_vs_implicit_open_text_for_coding.csv", index=False
    )

    with open(outdir / "README_explicit_implicit_comparison.txt", "w", encoding="utf-8") as f:
        f.write(
            "Explicit vs. implicit BiasAuditor survey comparison\\n\\n"
            "Files: explicit_vs_implicit_overall_comparison.csv, "
            "explicit_vs_implicit_task_comparison.csv, "
            "explicit_vs_implicit_perception_comparison.csv, "
            "explicit_vs_implicit_report_influence_distributions.csv, and "
            "explicit_vs_implicit_open_text_for_coding.csv.\\n\\n"
            "Interpretation notes:\\n"
            "- Mann–Whitney U treats the explicit and implicit survey samples as independent. "
            "If the same people completed both surveys, use participant-linked paired analyses instead.\\n"
            "- Validate that task names match across surveys before interpreting task-level differences.\\n"
            "- Confirm separate gold labels and line numbers for implicit examples before running. "
            "The defaults are copied from the explicit survey and may not fit the implicit code.\\n"
            "- Compare effect sizes and descriptive trends as well as p-values; small samples limit power.\\n"
            "- Perception comparisons use the exported means/SDs descriptively; the current output does "
            "not run inferential tests on individual Likert responses.\\n"
            "- Open-ended responses are combined for manual coding, not automatically thematically analyzed.\\n"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze explicit and implicit BiasAuditor survey exports."
    )
    parser.add_argument("survey_file", nargs="?", help="Single Qualtrics CSV/TSV export")
    parser.add_argument("--outdir", default="biasauditor_results", help="Output directory")
    parser.add_argument("--compare-explicit", help="Explicit survey CSV/TSV")
    parser.add_argument("--compare-implicit", help="Implicit survey CSV/TSV")
    parser.add_argument("--comparison-outdir", default="biasauditor_comparison",
                        help="Directory for cross-survey comparison outputs")
    args = parser.parse_args()

    if args.compare_explicit or args.compare_implicit:
        if not (args.compare_explicit and args.compare_implicit):
            parser.error("Provide both --compare-explicit and --compare-implicit.")
        script_path = Path(__file__).resolve()
        explicit_dir = Path(args.outdir) / "explicit"
        implicit_dir = Path(args.outdir) / "implicit"

        # Reuse the existing single-survey pipeline for each export.
        import subprocess
        explicit_env = os.environ.copy()
        explicit_env["BIASAUDITOR_SURVEY_TYPE"] = "explicit"
        implicit_env = os.environ.copy()
        implicit_env["BIASAUDITOR_SURVEY_TYPE"] = "implicit"
        subprocess.run(
            [sys.executable, str(script_path), args.compare_explicit,
             "--outdir", str(explicit_dir)], check=True, env=explicit_env
        )
        subprocess.run(
            [sys.executable, str(script_path), args.compare_implicit,
             "--outdir", str(implicit_dir)], check=True, env=implicit_env
        )
        compare_outputs(explicit_dir, implicit_dir, Path(args.comparison_outdir))
        print(f"Explicit/implicit comparison complete: {Path(args.comparison_outdir).resolve()}")
    else:
        if not args.survey_file:
            parser.error("Provide survey_file or both --compare-explicit and --compare-implicit.")
        # Existing single-survey analyzer accepts positional args from sys.argv.
        analyze_single()


if __name__ == "__main__":
    import sys
    main()
