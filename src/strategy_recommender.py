import numpy as np
import pandas as pd

def estimate_tyre_life(degradation_df):
    """
    Estimate average usable life (in laps) for each compound based on degradation.
    We assume the stint ends when lap time increases >5% from the first lap average.
    """
    tyre_life = {}
    for compound, grp in degradation_df.groupby("Compound"):
        base = grp["LapTime_s"].iloc[0]
        threshold = base * 1.05
        exceed = grp[grp["LapTime_s"] > threshold]
        if not exceed.empty:
            tyre_life[compound] = exceed["LapIndex"].iloc[0]
        else:
            tyre_life[compound] = grp["LapIndex"].max()
    return tyre_life


def simulate_strategy(total_laps, compound_sequence, degradation_df, pit_loss=20.0, tyre_life=None):
    """
    Simulate a race given a tyre compound sequence (list like ['M', 'H']) and degradation model.

    Parameters:
    - total_laps: number of laps in race
    - compound_sequence: e.g. ['M', 'H'] or ['S', 'M', 'H']
    - degradation_df: mean degradation data from average_degradation()
    - pit_loss: seconds lost during a pit stop

    Returns:
    - dict with total_race_time (s) and detailed stint times
    """

    if tyre_life is None:
        tyre_life = estimate_tyre_life(degradation_df)
    stint_results = []
    laps_remaining = total_laps
    total_time = 0.0

    for compound in compound_sequence:
        # max laps for this stint
        stint_len = min(int(tyre_life.get(compound, 10)), laps_remaining)
        avg_curve = degradation_df[degradation_df["Compound"] == compound]["LapTime_s"].values
        if len(avg_curve) == 0:
            avg_curve = np.linspace(90, 95, stint_len)  # fallback baseline

        # repeat or truncate the degradation curve
        if len(avg_curve) < stint_len:
            lap_times = np.interp(
                np.linspace(0, len(avg_curve)-1, stint_len),
                np.arange(len(avg_curve)),
                avg_curve
            )
        else:
            lap_times = avg_curve[:stint_len]

        stint_time = lap_times.sum()
        total_time += stint_time
        stint_results.append({
            "Compound": compound,
            "Laps": stint_len,
            "Time_s": stint_time
        })

        laps_remaining -= stint_len
        if laps_remaining <= 0:
            break
        total_time += pit_loss  # add pit loss between stints

    return {
        "total_race_time_s": total_time,
        "stints": stint_results,
        "sequence": compound_sequence
    }

def compute_candidate_pit_laps(total_laps, tyre_life):
    """
    Returns list of candidate pit lap numbers for each compound based on tyre_life mapping {compound: laps}.
    For example: if tyre_life['M']=15 and total_laps=60 -> candidate midpoints [15,30,45] etc.
    """
    candidates = set()
    for comp, life in tyre_life.items():
        step = max(1, life)
        lap = step
        while lap < total_laps:
            candidates.add(lap)
            lap += step
    return sorted(candidates)

def recommend_optimal_strategy(total_laps, degradation_df, pit_loss=20.0,
                               circuit_type="balanced", weather="dry",
                               qualifying_position=None):
    """
    Recommend the optimal pit strategy under realistic FIA and circuit conditions.

    Parameters:
    - total_laps: number of race laps
    - degradation_df: tyre degradation table
    - pit_loss: base pit loss (s)
    - circuit_type: 'high_deg', 'low_deg', or 'balanced'
    - weather: 'dry' or 'wet'
    - qualifying_position: int (1 = pole, higher = further back)

    Returns:
    - best strategy and ranked alternatives (DataFrame)
    """
    compound_options = degradation_df["Compound"].unique().tolist()
    results = []

    # --- Adjust degradation multipliers based on weather ---
    weather_factor = 1.0 if weather == "dry" else 0.7
    pit_loss *= 1.1 if weather == "wet" else 1.0

    # --- Estimate degradation life per compound ---
    tyre_life = estimate_tyre_life(degradation_df)
    for k in tyre_life:
        tyre_life[k] = int(tyre_life[k] * weather_factor)

    # --- Adjust stint count based on circuit type ---
    if circuit_type == "high_deg":
        stint_count_options = [2, 3]
    elif circuit_type == "low_deg":
        stint_count_options = [1, 2]
    else:
        stint_count_options = [1, 2]

    # --- Adjust for qualifying position bias ---
    if qualifying_position:
        if qualifying_position <= 3:
            # front runners -> favor track position
            stint_count_options = [1, 2]
            pit_loss *= 1.05  # conservative approach
        elif 4 <= qualifying_position <= 10:
            # midfield -> balanced
            stint_count_options = [2]
        else:
            # backmarkers -> aggressive, multiple stints
            stint_count_options = [2, 3]
            pit_loss *= 0.95  # more aggressive

    # --- Generate valid compound sequences ---
    for num_stints in stint_count_options:
        sequences = list(_generate_compound_sequences(compound_options, num_stints, weather))
        for seq in sequences:
            sim = simulate_strategy(total_laps, seq, degradation_df, pit_loss=pit_loss, tyre_life=tyre_life)
            results.append(sim)

    if not results:
        return None, pd.DataFrame()

    results_df = pd.DataFrame([
        {"Strategy": " → ".join(r["sequence"]), "TotalTime_s": r["total_race_time_s"]}
        for r in results
    ]).sort_values("TotalTime_s")

    best = results_df.iloc[0]
    return best, results_df

def _generate_compound_sequences(compounds, stint_count, weather):
    """
    Helper to generate only valid FIA-legal sequences:
    - In dry races, must use at least 2 different compounds.
    - In wet races, all wet or inter compounds are allowed freely.
    """
    import itertools

    if weather == "dry":
        all_sequences = list(itertools.product(compounds, repeat=stint_count))
        valid = [seq for seq in all_sequences if len(set(seq)) >= 2]
    else:
        valid = list(itertools.product(compounds, repeat=stint_count))

    return valid
