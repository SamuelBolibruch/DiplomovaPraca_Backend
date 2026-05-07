import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from xgboost import XGBClassifier

BASE_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "exp1_model_comparison")
RANDOM_STATE = 42

LEGIT_TRAIN = 10
LEGIT_TEST = 5

EXCLUDE_COLS = {"UserId", "RoundId", "label"}

DATASET_RUNS = [
    ("general", os.path.join(os.path.dirname(__file__), "..", "data", "training")),
    ("personal", os.path.join(os.path.dirname(__file__), "..", "data", "training_personal")),
]

os.makedirs(BASE_RESULTS_DIR, exist_ok=True)


def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["label"].values
    return X, y


def split_user_data(X, y, legit_train: int = LEGIT_TRAIN, legit_test: int = LEGIT_TEST):
    legit_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    n_legit = len(legit_indices)
    needed = legit_train + legit_test

    if n_legit < needed:
        return None

    rng = np.random.default_rng(RANDOM_STATE)
    perm_legit = rng.permutation(legit_indices)
    train_legit = perm_legit[:legit_train]
    test_legit = perm_legit[legit_train:legit_train + legit_test]

    n_neg = len(neg_indices)
    ratio = legit_train / needed
    n_neg_train = int(round(n_neg * ratio))
    n_neg_test = n_neg - n_neg_train

    perm_neg = rng.permutation(neg_indices)
    train_neg = perm_neg[:n_neg_train]
    test_neg = perm_neg[n_neg_train:n_neg_train + n_neg_test]

    train_idx = np.concatenate([train_legit, train_neg])
    test_idx = np.concatenate([test_legit, test_neg])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_arr = X.values
    return (
        X_arr[train_idx], X_arr[test_idx],
        y[train_idx], y[test_idx],
    )


def find_best_threshold_from_scores(oof_probs, y_train):
    candidates = np.linspace(0.0, 1.0, 201)
    best_threshold = 0.5
    best_diff = float("inf")

    for thr in candidates:
        preds = (oof_probs >= thr).astype(int)
        tn, fp, fn, tp = _safe_confusion(y_train, preds)
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            best_threshold = thr

    return best_threshold


def get_oof_probabilities(model_name: str, X_train, y_train, n_splits: int = 5):
    classes, counts = np.unique(y_train, return_counts=True)
    min_count = int(counts.min())

    effective_splits = min(n_splits, min_count)
    if effective_splits < 2:
        warnings.warn(
            f"[OOF] Model {model_name}: nedostatok vzoriek (min trieda má {min_count}) "
            f"pre StratifiedKFold. OOF nie je možné."
        )
        return None

    if effective_splits < n_splits:
        warnings.warn(
            f"[OOF] Model {model_name}: n_splits znížený z {n_splits} na {effective_splits} "
            f"(min trieda má {min_count} vzoriek)."
        )

    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = np.zeros(len(y_train), dtype=float)

    for train_idx, val_idx in skf.split(X_train, y_train):
        fold_model = get_model(model_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fold_model.fit(X_train[train_idx], y_train[train_idx])
        oof_probs[val_idx] = fold_model.predict_proba(X_train[val_idx])[:, 1]

    return oof_probs


def compute_eer(far: float, frr: float):
    return (far + frr) / 2.0


def _safe_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def compute_metrics(y_true, y_pred, _probs):
    tn, fp, fn, tp = _safe_confusion(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    eer = compute_eer(far, frr)
    return {"accuracy": acc, "far": far, "frr": frr, "eer": eer}


def get_model(model_name: str):
    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    elif model_name == "SVM":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ])

    elif model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            verbosity=0,
        )

    else:
        raise ValueError(f"Neznámy model: {model_name}")


def train_and_evaluate_model(model_name: str, X_train, X_test, y_train, y_test):
    oof_probs = get_oof_probabilities(model_name, X_train, y_train, n_splits=5)
    if oof_probs is None:
        return None

    threshold = find_best_threshold_from_scores(oof_probs, y_train)

    model = get_model(model_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    probs_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (probs_test >= threshold).astype(int)

    metrics = compute_metrics(y_test, y_pred_test, probs_test)
    metrics["threshold"] = threshold
    return metrics


def print_dataset_summary(
    dataset_name: str,
    all_user_ids,
    per_user_df,
    final_table,
    summary_df,
    ranking_df,
    skipped_for_samples,
    skipped_oof_records,
    combined_plot,
):
    print("\n" + "=" * 80)
    print(f"DATASET: {dataset_name.upper()}")
    print("=" * 80)
    print("SÚHRN – priemery a smerodajné odchýlky cez všetkých používateľov")
    print("=" * 80)
    header = f"{'Model':<16} {'Acc':>8} {'±':>6} {'FAR':>8} {'±':>6} {'FRR':>8} {'±':>6} {'EER':>8} {'±':>6}"
    print(header)
    print("-" * 80)
    for _, row in summary_df.iterrows():
        print(
            f"{row['model']:<16}"
            f" {row['avg_accuracy']:>8.4f} {row['std_accuracy']:>6.4f}"
            f" {row['avg_far']:>8.4f} {row['std_far']:>6.4f}"
            f" {row['avg_frr']:>8.4f} {row['std_frr']:>6.4f}"
            f" {row['avg_eer']:>8.4f} {row['std_eer']:>6.4f}"
        )
    print("=" * 80)

    print("\nA) Hlavná súhrnná tabuľka po modeloch")
    print("-" * 80)
    print(final_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nB) Poradie modelov podľa EER (najlepší -> najhorší)")
    print("-" * 80)
    print(ranking_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nC) Počet hodnotených používateľov")
    print("-" * 80)
    evaluated_users = sorted(per_user_df["user_id"].unique().tolist())
    excluded_users = sorted(set(all_user_ids) - set(evaluated_users))
    print(f"Nájdených training súborov: {len(all_user_ids)}")
    print(f"Reálne zahrnutých používateľov: {len(evaluated_users)}")
    print(f"Vyradených používateľov: {len(excluded_users)}")

    if skipped_for_samples:
        print("Používatelia vyradení pre nedostatok legitímnych vzoriek:")
        for rec in skipped_for_samples:
            print(f"  - {rec['user_id']}: legit={rec['n_legit']}")
    else:
        print("Nikto nebol vyradený pre nedostatok legitímnych vzoriek.")

    if skipped_oof_records:
        print("Preskočené kombinácie user/model pre OOF:")
        for rec in skipped_oof_records:
            print(f"  - {rec['user_id']} | {rec['model']}")
    else:
        print("Žiadna kombinácia user/model nebola preskočená kvôli OOF.")

    print("\nD) Potvrdenie metodiky")
    print("-" * 80)
    print("- Model sa trénuje samostatne pre každého používateľa.")
    print("- Výsledky sa agregujú priemerom a smerodajnou odchýlkou cez používateľov.")
    print("- Threshold sa určuje user-specific z OOF predikcií na train množine.")

    print("\nE) Grafy vhodné do diplomovej práce")
    print("-" * 80)
    print(f"- metrics_comparison_{dataset_name}.png: {combined_plot}")

    print("\nF) Krátke textové zhrnutie")
    print("-" * 80)
    best_model = ranking_df.iloc[0]["Model"]
    best_eer = ranking_df.iloc[0]["EER_avg"]
    second_model = ranking_df.iloc[1]["Model"]
    second_eer = ranking_df.iloc[1]["EER_avg"]
    worst_model = ranking_df.iloc[-1]["Model"]
    worst_eer = ranking_df.iloc[-1]["EER_avg"]
    print(
        f"Najlepší model podľa EER je {best_model} (EER={best_eer:.4f}). "
        f"Druhý je {second_model} (EER={second_eer:.4f}) a najhorší je {worst_model} "
        f"(EER={worst_eer:.4f})."
    )
    print(
        "Rozdiely medzi modelmi sú priamo viditeľné v tabuľke aj v grafe "
        "AAR/FAR/FRR/EER s error bars."
    )
    print(
        "V texte diplomovej práce je možné uviesť poradie modelov podľa EER a "
        "zdôrazniť, že hodnotenie bolo robené user-specific spôsobom."
    )


def run_experiment(dataset_name: str, training_dir: str):
    results_dir = os.path.join(BASE_RESULTS_DIR, dataset_name)
    os.makedirs(results_dir, exist_ok=True)

    model_names = ["RandomForest", "SVM", "XGBoost"]

    csv_files = sorted(
        f for f in os.listdir(training_dir)
        if f.startswith("training_") and f.endswith(".csv")
    )

    all_user_ids = [f.replace("training_", "").replace(".csv", "") for f in csv_files]
    skipped_for_samples = []
    skipped_oof_records = []

    print("\n" + "#" * 80)
    print(f"Experiment 1 | dataset: {dataset_name} | directory: {training_dir}")
    print("#" * 80)
    print(f"Nájdených {len(csv_files)} training súborov.\n")

    per_user_records = []

    for csv_file in csv_files:
        uid = csv_file.replace("training_", "").replace(".csv", "")
        csv_path = os.path.join(training_dir, csv_file)

        X, y = load_dataset(csv_path)

        split = split_user_data(X, y)
        if split is None:
            n_legit = int((y == 1).sum())
            warnings.warn(
                f"[SKIP] Dataset {dataset_name}, user {uid}: iba {n_legit} legitímnych vzoriek "
                f"(potrebných {LEGIT_TRAIN + LEGIT_TEST}). Používateľ preskočený."
            )
            skipped_for_samples.append({"user_id": uid, "n_legit": n_legit})
            continue

        X_train, X_test, y_train, y_test = split

        print(f"User: {uid}  |  train: {len(y_train)} vzoriek  |  test: {len(y_test)} vzoriek")

        for model_name in model_names:
            metrics = train_and_evaluate_model(model_name, X_train, X_test, y_train, y_test)

            if metrics is None:
                warnings.warn(
                    f"[SKIP] Dataset {dataset_name}, user {uid}, model {model_name}: "
                    "OOF zlyhalo, výsledky preskočené."
                )
                skipped_oof_records.append({"user_id": uid, "model": model_name})
                continue

            record = {
                "user_id": uid,
                "model": model_name,
                "threshold": metrics["threshold"],
                "accuracy": metrics["accuracy"],
                "far": metrics["far"],
                "frr": metrics["frr"],
                "eer": metrics["eer"],
            }
            per_user_records.append(record)
            print(
                f"  {model_name:14s}  acc={metrics['accuracy']:.3f}  "
                f"FAR={metrics['far']:.3f}  FRR={metrics['frr']:.3f}  "
                f"EER={metrics['eer']:.3f}  thr={metrics['threshold']:.3f}"
            )

        print()

    if not per_user_records:
        print(f"Žiadne výsledky neboli vytvorené pre dataset {dataset_name}.")
        return None

    per_user_df = pd.DataFrame(per_user_records)
    per_user_csv = os.path.join(results_dir, f"per_user_results_{dataset_name}.csv")
    per_user_df.to_csv(per_user_csv, index=False)
    print(f"Per-user výsledky uložené do: {per_user_csv}\n")

    summary_records = []
    for model_name in model_names:
        subset = per_user_df[per_user_df["model"] == model_name]
        record = {"model": model_name}
        for metric in ["accuracy", "far", "frr", "eer"]:
            record[f"avg_{metric}"] = subset[metric].mean()
            record[f"std_{metric}"] = subset[metric].std()
        summary_records.append(record)

    summary_df = pd.DataFrame(summary_records)
    summary_csv = os.path.join(results_dir, f"summary_results_{dataset_name}.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Súhrnné výsledky uložené do: {summary_csv}\n")

    final_table = summary_df.rename(
        columns={
            "model": "Model",
            "avg_accuracy": "Acc_avg",
            "std_accuracy": "Acc_std",
            "avg_far": "FAR_avg",
            "std_far": "FAR_std",
            "avg_frr": "FRR_avg",
            "std_frr": "FRR_std",
            "avg_eer": "EER_avg",
            "std_eer": "EER_std",
        }
    )[[
        "Model", "Acc_avg", "Acc_std", "FAR_avg", "FAR_std",
        "FRR_avg", "FRR_std", "EER_avg", "EER_std"
    ]]

    ranking_df = final_table[["Model", "EER_avg"]].sort_values("EER_avg", ascending=True)

    metrics_to_plot = ["avg_accuracy", "avg_far", "avg_frr", "avg_eer"]
    metric_labels = ["AAR (Accuracy)", "FAR", "FRR", "EER"]
    metric_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    metric_std = ["std_accuracy", "std_far", "std_frr", "std_eer"]

    n_models = len(summary_df)
    n_metrics = len(metrics_to_plot)
    bar_width = 0.18
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(11, 6.5))

    for i, (metric, label, color, std_col) in enumerate(
        zip(metrics_to_plot, metric_labels, metric_colors, metric_std)
    ):
        offsets = x + (i - (n_metrics - 1) / 2) * bar_width
        bars = ax.bar(
            offsets, summary_df[metric], bar_width,
            label=label, color=color,
            yerr=summary_df[std_col], capsize=3,
        )
        for j, bar in enumerate(bars):
            h = bar.get_height()
            err = float(summary_df[std_col].iloc[j])
            y_text = h + err + 0.02
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_text,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
                clip_on=False,
            )

    ax.set_title(f"Experiment 1 ({dataset_name}) – porovnanie modelov: AAR, FAR, FRR, EER")
    ax.set_ylabel("Hodnota")
    ax.set_xlabel("Model")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["model"])
    y_max_data = float((summary_df[metrics_to_plot].values + summary_df[metric_std].values).max())
    ax.set_ylim(0, max(1.15, y_max_data + 0.12))
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")

    plt.tight_layout()
    combined_plot = os.path.join(results_dir, f"metrics_comparison_{dataset_name}.png")
    plt.savefig(combined_plot, dpi=150)
    plt.close()
    print(f"\nGraf metrík uložený do: {combined_plot}")

    print_dataset_summary(
        dataset_name=dataset_name,
        all_user_ids=all_user_ids,
        per_user_df=per_user_df,
        final_table=final_table,
        summary_df=summary_df,
        ranking_df=ranking_df,
        skipped_for_samples=skipped_for_samples,
        skipped_oof_records=skipped_oof_records,
        combined_plot=combined_plot,
    )

    return {
        "dataset_name": dataset_name,
        "final_table": final_table,
        "ranking_df": ranking_df,
        "results_dir": results_dir,
    }


def main():
    run_summaries = []
    for dataset_name, training_dir in DATASET_RUNS:
        run_summary = run_experiment(dataset_name, training_dir)
        if run_summary is not None:
            run_summaries.append(run_summary)

    if not run_summaries:
        print("Žiadny dataset nevytvoril výsledky.")
        return

    print("\n" + "#" * 80)
    print("Experiment 1 | finálne porovnanie datasetov")
    print("#" * 80)
    for run_summary in run_summaries:
        ranking_df = run_summary["ranking_df"]
        best_model = ranking_df.iloc[0]["Model"]
        best_eer = ranking_df.iloc[0]["EER_avg"]
        print(
            f"Dataset {run_summary['dataset_name']}: najlepší model = {best_model} "
            f"(EER={best_eer:.4f}) | výstupy: {run_summary['results_dir']}"
        )

    combined_final_tables = []
    for run_summary in run_summaries:
        dataset_table = run_summary["final_table"].copy()
        dataset_table.insert(0, "Dataset", run_summary["dataset_name"])
        combined_final_tables.append(dataset_table)

    combined_df = pd.concat(combined_final_tables, ignore_index=True)
    combined_csv = os.path.join(BASE_RESULTS_DIR, "final_summary_table_all_datasets.csv")
    combined_df.to_csv(combined_csv, index=False)

    print("\nG) Spoločná tabuľka pre všetky datasety")
    print("-" * 80)
    print(combined_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Spoločná tabuľka uložená do: {combined_csv}")


if __name__ == "__main__":
    main()
