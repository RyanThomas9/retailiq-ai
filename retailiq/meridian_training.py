
def truth_roi(truth_csv, spend_csv, outcome, weekly_csv):
    """
    Calculate true simulated iROAS separately for Stores and E-commerce.

    Ground truth is used only for model validation.
    It must never be supplied to Meridian during training.
    """
    if outcome not in OUTCOMES:
        raise ValueError(f"Unknown outcome: {outcome}")

    truth = pd.read_csv(
        truth_csv,
        parse_dates=["week_start_date"]
    )

    spend = pd.read_csv(
        spend_csv,
        parse_dates=["week_start_date"]
    )

    # Retained for compatibility with the existing training script.
    weekly = pd.read_csv(
        weekly_csv,
        parse_dates=["week_start_date"]
    )

    # Validate the expected columns.
    required_truth = {
        "week_start_date",
        "subchannel",
        "outcome",
        "true_incremental_revenue"
    }

    missing = required_truth - set(truth.columns)
    if missing:
        raise ValueError(
            f"Missing ground-truth columns: {sorted(missing)}"
        )

    required_spend = {
        "week_start_date",
        "subchannel",
        "spend"
    }

    missing = required_spend - set(spend.columns)
    if missing:
        raise ValueError(
            f"Missing spend columns: {sorted(missing)}"
        )

    # Select the true marketing effects for this outcome.
    truth = truth.loc[
        truth["outcome"] == outcome
    ].copy()

    if truth.empty:
        raise ValueError(
            f"No ground-truth records for {outcome}"
        )

    # Ensure ground truth and spend cover the same weeks.
    truth_weeks = set(truth["week_start_date"])
    spend_weeks = set(spend["week_start_date"])

    if truth_weeks != spend_weeks:
        raise ValueError(
            "Ground-truth and spend periods do not match"
        )

    # Sum actual simulated incremental revenue by subchannel.
    incremental = (
        truth.groupby("subchannel")["true_incremental_revenue"]
        .sum()
    )

    # Marketing spend is shared across both outcome models.
    total_spend = (
        spend.groupby("subchannel")["spend"]
        .sum()
    )

    result = pd.DataFrame({
        "true_incremental_revenue": incremental,
        "total_spend": total_spend
    })

    if result.isna().any().any():
        raise ValueError(
            "Ground-truth and spend subchannels do not match"
        )

    if (result["total_spend"] <= 0).any():
        raise ValueError(
            "Marketing spend must be positive"
        )

    result["true_roi"] = (
        result["true_incremental_revenue"]
        / result["total_spend"]
    )

    return result[["true_roi"]].reset_index()
