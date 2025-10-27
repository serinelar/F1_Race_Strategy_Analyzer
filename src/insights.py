# src/insights.py
import numpy as np
import pandas as pd

def lap_time_insights(stints_df, top_n_compounds=3):
    """
    stints_df: DataFrame with columns: Driver, LapNumber, Compound, LapTime_s, Stint
    Returns list of short insight strings.
    """
    if stints_df is None or stints_df.empty:
        return ["No lap/time data available to generate insights."]

    out = []
    # 1) detect number of stints (per driver)
    stint_counts = stints_df.groupby('Driver')['Stint'].nunique()
    most_stints_driver = stint_counts.idxmax() if not stint_counts.empty else None
    if most_stints_driver:
        out.append(f"{most_stints_driver} had the highest number of stints ({stint_counts.max()}).")

    # 2) detect lap-time spikes (pit laps / incidents)
    # compute lap-to-lap diff per driver
    spikes = []
    for driver, g in stints_df.groupby('Driver'):
        g = g.sort_values('LapNumber')
        if 'LapTime_s' not in g.columns or g['LapTime_s'].isna().all():
            continue
        diffs = g['LapTime_s'].diff().abs().fillna(0)
        big = diffs[diffs > diffs.mean() + 2 * diffs.std()]
        if not big.empty:
            spikes.append((driver, int(big.idxmax())))
    if spikes:
        sample = spikes[:3]
        txt = ", ".join([f"{d} (spike at idx {i})" for d, i in sample])
        out.append(f"Lap-time spikes detected for: {txt}. These usually indicate pit-laps or traffic/incidents.")

    # 3) degradation slope per compound
    if 'Compound' in stints_df.columns:
        deg = stints_df.groupby(['Compound', 'LapNumber'], as_index=False)['LapTime_s'].mean()
        slopes = {}
        for c, g in deg.groupby('Compound'):
            if len(g) >= 2:
                x = g['LapNumber'].values
                y = g['LapTime_s'].values
                slope = np.polyfit(x, y, 1)[0]
                slopes[c] = slope
        if slopes:
            # sort by slope descending (fastest degradation first)
            sorted_s = sorted(slopes.items(), key=lambda t: -t[1])
            top = ", ".join([f"{c} (slope {s:.4f}s/lap)" for c, s in sorted_s[:top_n_compounds]])
            out.append(f"Estimated degradation slope (s per lap): {top}. Higher slope = faster wear.")

    # 4) recommended focus
    out.append("Recommendation: inspect pit lap timing and tyre-change laps; use the tyre with the flattest slope for longer stints.")

    return out
