import numpy as np

from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score


def returns_on_predictions(df, y_preds):
    def _classify_prediction(example):
        true_label = example["label"]
        prediction = example["prediction"]

        if true_label == 1 and prediction == 1:
            return "TP"
        elif true_label == 1 and prediction == 0:
            return "FN"
        elif true_label == 0 and prediction == 0:
            return "TN"
        elif true_label == 0 and prediction == 1:
            return "FP"

    df["prediction"] = y_preds
    df["result"] = df.apply(_classify_prediction, axis=1)

    print("\nTotals:")
    print(df[["result", "prediction"]].groupby("result").count())

    print("\nMeans:")
    print(
        df[
            [
                "result",
                "return_one_month",
                "return_two_month",
                "return_three_month",
            ]
        ]
        .groupby("result")
        .mean()
    )

    print("\nStds:")
    print(
        df[
            [
                "result",
                "return_one_month",
                "return_two_month",
                "return_three_month",
            ]
        ]
        .groupby("result")
        .std()
    )

    print("\nPositive predictions:")
    print("\nMeans:")
    print(
        df[df.prediction == 1][
            ["return_one_month", "return_two_month", "return_three_month"]
        ].mean()
    )

    print("\nStds:")
    print(
        df[df.prediction == 1][
            ["return_one_month", "return_two_month", "return_three_month"]
        ].std()
    )

    return df


def performance_summary(y_score, y_preds, y_true, auc_cutoff=0.6):
    precision = precision_score(y_true, y_preds, zero_division=0)
    avg_precision = average_precision_from_cutoff(y_true, y_score, auc_cutoff)
    if np.any(y_true):
        roc = roc_auc_score(y_true, y_score)
    else:
        roc = 0

    print("Precision:", precision)
    print("PR-AUC/AP score:", avg_precision)
    print("ROC-AUC score:", roc)
    print("Total positive predictions:", y_preds.sum())


def performance_on_slice(df, y_score, y_preds, on_slice, top_ten):
    def _performance_summary(_df):
        print("\n", _df[on_slice].iloc[0], ":")
        if top_ten:
            top_scores = _df.sort_values("score", ascending=False).iloc[:10]
            top_scores["prediction"] = np.ones(len(top_scores))
            performance_summary(
                top_scores["score"],
                top_scores["prediction"],
                top_scores["label"],
            )
        else:
            performance_summary(_df["score"], _df["prediction"], _df["label"])

    df["prediction"] = y_preds
    df["score"] = y_score

    df[[on_slice, "prediction", "score", "label"]].groupby(on_slice).apply(
        _performance_summary
    )


def average_precision_from_cutoff(y_true, y_score, cutoff):
    # precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    selected_preds = y_score >= cutoff
    if selected_preds.sum() == 0:
        return 0
    return average_precision_score(y_true[selected_preds], y_score[selected_preds])
