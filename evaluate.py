import numpy as np
import pandas as pd

from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score


def returns_on_predictions(df, verbose=False):
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

    evaluation_metrics = {}
    df["result"] = df.apply(_classify_prediction, axis=1)

    if verbose:
        print("\nTotals:")
        print(df[["result", "prediction"]].groupby("result").count())
        print("\nMeans:")
    mean_returns = (
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

    if verbose:
        print(mean_returns)
    evaluation_metrics["_fp_ret_1mo"] = np.round(
        mean_returns["return_one_month"].get("FP", 0), 3
    )
    evaluation_metrics["_fp_ret_2mo"] = np.round(
        mean_returns["return_two_month"].get("FP", 0), 3
    )
    evaluation_metrics["_fp_ret_3mo"] = np.round(
        mean_returns["return_three_month"].get("FP", 0), 3
    )

    if verbose:
        print("\nStd:")
    mean_returns_stds = (
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
    if verbose:
        print(mean_returns_stds)
    evaluation_metrics["_fp_ret_std_1mo"] = np.round(
        mean_returns_stds["return_one_month"].get("FP", 0), 3
    )
    evaluation_metrics["_fp_ret_std_2mo"] = np.round(
        mean_returns_stds["return_two_month"].get("FP", 0), 3
    )
    evaluation_metrics["_fp_ret_std_3mo"] = np.round(
        mean_returns_stds["return_three_month"].get("FP", 0), 3
    )

    if verbose:
        print("\nPositive predictions:")
        print("\nMeans:")
    positive_preds_means = df[df.prediction == 1][
        ["return_one_month", "return_two_month", "return_three_month"]
    ].mean()
    if verbose:
        print(positive_preds_means)
    evaluation_metrics["_pos_pred_ret_1mo"] = np.round(
        positive_preds_means["return_one_month"], 3
    )
    evaluation_metrics["_pos_pred_ret_2mo"] = np.round(
        positive_preds_means["return_two_month"], 3
    )
    evaluation_metrics["_pos_pred_ret_3mo"] = np.round(
        positive_preds_means["return_three_month"], 3
    )

    if verbose:
        print("\nStd:")
    positive_preds_stds = df[df.prediction == 1][
        ["return_one_month", "return_two_month", "return_three_month"]
    ].std()
    if verbose:
        print(positive_preds_stds)
    evaluation_metrics["_pos_pred_ret_std_1mo"] = np.round(
        positive_preds_stds["return_one_month"], 3
    )
    evaluation_metrics["_pos_pred_ret_std_2mo"] = np.round(
        positive_preds_stds["return_two_month"], 3
    )
    evaluation_metrics["_pos_pred_ret_std_3mo"] = np.round(
        positive_preds_stds["return_three_month"], 3
    )

    return df, pd.Series(evaluation_metrics)


def performance_summary(
    y_score: np.array,
    y_preds: np.array,
    y_true: np.array,
    auc_cutoff: float,
    verbose=False,
) -> pd.Series:
    evaluation_metrics = {}
    precision = precision_score(y_true, y_preds, zero_division=0)
    avg_precision = average_precision_from_cutoff(y_true, y_score, auc_cutoff)
    if np.any(y_true):
        roc = roc_auc_score(y_true, y_score)
    else:
        roc = 0

    if verbose:
        print("Precision:", precision)
        print("PR-AUC/AP score:", avg_precision)
        print("ROC-AUC score:", roc)
        print("Total positive predictions:", y_preds.sum())
        print("Total positive labels:", y_true.sum())

    evaluation_metrics["_precision"] = np.round(precision, 3)
    evaluation_metrics["_ap"] = np.round(avg_precision, 3)
    evaluation_metrics["_roc"] = np.round(roc, 3)
    evaluation_metrics["_positive_preds"] = int(y_preds.sum())
    evaluation_metrics["_positive_labels"] = int(y_true.sum())

    return pd.Series(evaluation_metrics)


def performance_on_slice(df: pd.DataFrame, on_slice: str, auc_cutoff=0.5) -> None:
    def _performance_summary(_df):
        print("\n", _df[on_slice].iloc[0], ":")
        performance_summary(_df["score"], _df["prediction"], _df["label"], auc_cutoff)

    df[[on_slice, "prediction", "score", "label"]].groupby(on_slice).apply(
        _performance_summary
    )


def average_precision_from_cutoff(
    y_true: np.array, y_score: np.array, cutoff: float
) -> float:
    # precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    selected_preds = y_score >= cutoff
    if selected_preds.sum() == 0:
        return 0
    return average_precision_score(y_true[selected_preds], y_score[selected_preds])


def performance_on_trading_use_case(
    df: pd.DataFrame, top_n_trades: int, min_industry_score: float
) -> pd.Series:
    evaluation_metrics = {}

    for name, group in df[
        [
            "ticker_x",
            "ticker_y",
            "subindustry",
            "score",
            "label",
            "return_one_month",
            "return_two_month",
            "return_three_month",
        ]
    ].groupby("subindustry"):
        total_trades = len(group)
        if total_trades > 0:
            selected_group = (
                group[group.score >= min_industry_score]
                .sort_values(by="score", ascending=False)
                .drop_duplicates(subset="ticker_x", keep="first")
                .drop_duplicates(subset="ticker_y", keep="first")
                .iloc[:top_n_trades]
            )
            short_name = name[:5] + name[-5:]
            selected_trades = len(selected_group)

            evaluation_metrics["{}_ntrades".format(short_name)] = int(selected_trades)

            evaluation_metrics[
                "{}_top{}_ret_1mo".format(short_name, top_n_trades)
            ] = np.round(selected_group["return_one_month"].mean(), 3)
            evaluation_metrics[
                "{}_top{}_ret_2mo".format(short_name, top_n_trades)
            ] = np.round(selected_group["return_two_month"].mean(), 3)
            evaluation_metrics[
                "{}_top{}_ret_3mo".format(short_name, top_n_trades)
            ] = np.round(selected_group["return_three_month"].mean(), 3)

            evaluation_metrics[
                "{}_top{}_ret_std_1mo".format(short_name, top_n_trades)
            ] = np.round(selected_group["return_one_month"].std(), 3)
            evaluation_metrics[
                "{}_top{}_ret_std_2mo".format(short_name, top_n_trades)
            ] = np.round(selected_group["return_two_month"].std(), 3)
            evaluation_metrics[
                "{}_top{}_ret_std_3mo".format(short_name, top_n_trades)
            ] = np.round(selected_group["return_three_month"].std(), 3)

    return pd.Series(evaluation_metrics)
