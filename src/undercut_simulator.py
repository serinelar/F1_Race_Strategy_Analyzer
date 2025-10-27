import numpy as np
import pandas as pd
import plotly.graph_objects as go

def _detect_laptime_column(df: pd.DataFrame):
    # prefer LapTime_s, then LapTime(s)
    for col in ["LapTime_s", "LapTime(s)", "LapTime"]:
        if col in df.columns:
            return col
    return None

def simulate_undercut_vs_overcut(degradation_df, compound_a="Medium", compound_b="Hard",
                                 undercut_lap=15, total_laps=60, pit_loss=20):
    """
    Simulate undercut vs overcut payoff between two drivers using different tyre compounds.

    degradation_df: DataFrame with columns ['Compound', 'LapIndex', 'LapTime_s' or 'LapTime(s)']
    """

    if degradation_df is None or degradation_df.empty:
        raise ValueError("degradation_df is empty. Need mean degradation per compound (Compound, LapIndex, LapTime).")

    laptime_col = _detect_laptime_column(degradation_df)
    if laptime_col is None:
        raise KeyError("degradation_df must contain a lap time column named one of: "
                       "'LapTime_s', 'LapTime(s)', or 'LapTime'.")

    # determine total_laps if not provided or too large
    max_index = int(degradation_df['LapIndex'].max()) if 'LapIndex' in degradation_df.columns else None
    if max_index is not None and (total_laps is None or total_laps > max_index):
        total_laps = max_index

    if total_laps <= 0:
        raise ValueError("total_laps must be > 0")

    # --- Prepare degradation profiles ---
    def get_deg_curve(comp):
        df = degradation_df[degradation_df["Compound"] == comp]
        if df.empty:
            raise ValueError(f"No degradation data for compound '{comp}'. Available: {degradation_df['Compound'].unique().tolist()}")
        # interpolate to full race length
        x_src = df["LapIndex"].values
        y_src = df[laptime_col].values.astype(float)
        x_target = np.arange(1, total_laps + 1)
        # if source has only one point, repeat it
        if len(x_src) == 1:
            return np.repeat(y_src[0], total_laps)
        return np.interp(x_target, x_src, y_src)

    laps_a = get_deg_curve(compound_a)
    laps_b = get_deg_curve(compound_b)

    # --- Simulate gap evolution ---
    gap = np.zeros(total_laps)
    delta_laps = []

    # use 0-based index for arrays but lap numbers are 1..total_laps
    for i in range(total_laps):
        lap_num = i + 1
        if lap_num < undercut_lap:
            # Before pit: both on their first stints (compare raw lap times)
            delta = laps_b[i] - laps_a[i]
        elif lap_num == undercut_lap:
            # Undercut pits now: instant pit delta (negative advantage for the pitting car)
            delta = -pit_loss
        else:
            # After undercut: undercut car on fresher tyres
            tyre_age = lap_num - undercut_lap
            # Simple model: fresh lap equals first-lap time of that compound, then small linear fade
            fresh_base = laps_a[0]
            # decay rate small (0.2% per lap) — tunable
            fresh_lap = fresh_base * (1 + 0.002 * tyre_age)
            delta = laps_b[i] - fresh_lap

        gap[i] = (gap[i - 1] + delta) if i > 0 else delta
        delta_laps.append(delta)

    payoff_df = pd.DataFrame({
        "Lap": np.arange(1, total_laps + 1),
        "Gap_to_Leader(s)": gap,
        "Delta(s)": delta_laps
    })

    # Identify undercut payoff lap
    min_gap = payoff_df["Gap_to_Leader(s)"].min()
    best_row = payoff_df.loc[payoff_df["Gap_to_Leader(s)"].idxmin()]
    best_lap = int(best_row["Lap"])

    payoff_summary = {
        "BestLap": best_lap,
        "MinGap(s)": float(min_gap),
        "Compound_A": compound_a,
        "Compound_B": compound_b
    }

    return payoff_df, payoff_summary


def plot_undercut_simulation(payoff_df, payoff_summary):
    """Visualize the undercut/overcut gap evolution."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=payoff_df["Lap"], y=payoff_df["Gap_to_Leader(s)"],
        mode="lines+markers",
        name="Gap Evolution",
        line=dict(width=3)
    ))

    # vertical line safe-guard: only draw if BestLap within range
    best_lap = payoff_summary.get("BestLap", None)
    if best_lap is not None and 1 <= best_lap <= payoff_df["Lap"].max():
        fig.add_vline(
            x=best_lap,
            line=dict(dash="dash", color="red"),
            annotation_text=f"Optimal Undercut Lap: {best_lap}",
            annotation_position="top left"
        )

    fig.update_layout(
        title=f"Undercut vs Overcut Payoff Simulation ({payoff_summary.get('Compound_A')} vs {payoff_summary.get('Compound_B')})",
        xaxis_title="Lap",
        yaxis_title="Gap to Leader (s)",
        template="plotly_dark",
        showlegend=False
    )
    return fig
