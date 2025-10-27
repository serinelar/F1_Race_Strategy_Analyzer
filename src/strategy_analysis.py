

import pandas as pd
import numpy as np

def compute_stint_data(laps):
    """
    Return laps with useful columns:
     - Driver (short name), LapNumber, Compound, Stint, LapTime (Timedelta), LapTime_s
     - Position and cumulative RaceTime if present.
    Expect laps from fastf1.Session.laps
    """
    if laps is None or laps.empty:
        return pd.DataFrame()

    df = laps.copy()
    # standardize columns names used below (FastF1 uses Driver, Stint, Compound, LapNumber, LapTime, Time, Position)
    # ensure LapTime is a Timedelta
    if 'LapTime' in df.columns:
        df = df.dropna(subset=['LapTime']).copy()
        df['LapTime_s'] = df['LapTime'].dt.total_seconds()
    else:
        df['LapTime_s'] = np.nan

    # compute cumulative race time (relative) if Time exists
    if 'Time' in df.columns:
        df = df.sort_values(['LapNumber', 'Stint'])
        # FastF1 laps.Time is absolute timestamps per lap start; compute cumulative per driver
        df['RaceTime_s'] = df.groupby('Driver')['LapTime_s'].cumsum()
    else:
        df['RaceTime_s'] = df.groupby('Driver')['LapTime_s'].cumsum()

    return df

def average_degradation(stints):
    """
    Compute mean lap time evolution per stint (to visualize tyre degradation)
    """
    degradation = (
        stints.groupby(['Driver', 'Compound', 'Stint'])
        .apply(lambda x: x.assign(LapIndex=range(1, len(x) + 1)))
        .reset_index(drop=True)
    )

    mean_degradation = (
        degradation.groupby(['Compound', 'LapIndex'], as_index=False)
        .agg({'LapTime_s': 'mean'})
    )
    return mean_degradation

def compute_gap_to_leader(stints):
    """
    Compute gap-to-leader per lap: for each lap number, compute leader's cumulative race time
    and get other drivers' gap in seconds.
    """
    if stints.empty:
        return pd.DataFrame()

    # need RaceTime_s and LapNumber
    if 'RaceTime_s' not in stints.columns or 'LapNumber' not in stints.columns:
        raise ValueError("stints must contain 'RaceTime_s' and 'LapNumber'")

    # leader per lap = min RaceTime_s among drivers for that lap
    leader = stints.groupby('LapNumber').apply(lambda g: g.loc[g['RaceTime_s'].idxmin()][['RaceTime_s']])
    leader = leader.rename(columns={'RaceTime_s': 'LeaderTime_s'}).reset_index()
    merged = stints.merge(leader, on='LapNumber', how='left')
    merged['GapToLeader_s'] = merged['RaceTime_s'] - merged['LeaderTime_s']
    return merged

def detect_pit_stops_from_compound(laps):
    """
    Estimate pit events by detecting when a driver's compound value changes between laps.
    Returns DataFrame of pit events per driver with pit_lap (the lap when change occurred),
    previous_compound, next_compound.
    """
    if laps.empty or 'Compound' not in laps.columns:
        return pd.DataFrame(columns=['Driver', 'pit_lap', 'prev_compound', 'next_compound'])

    laps_sorted = laps.sort_values(['Driver', 'LapNumber']).copy()
    laps_sorted['prev_compound'] = laps_sorted.groupby('Driver')['Compound'].shift(1)
    # pit detected when prev_compound != Compound
    pit_events = laps_sorted[laps_sorted['prev_compound'].notna() & (laps_sorted['prev_compound'] != laps_sorted['Compound'])].copy()
    pit_events = pit_events[['Driver', 'LapNumber', 'prev_compound', 'Compound']].rename(
        columns={'LapNumber': 'pit_lap', 'Compound': 'next_compound'}
    ).reset_index(drop=True)
    return pit_events

def simulate_undercut_overtake(laps, driver, rival, candidate_pit_lap, pit_delta_est=20.0):
    """
    Simple undercut/overcut simulator.

    Inputs:
    - laps: DataFrame with LapNumber, Driver, LapTime_s, RaceTime_s
    - driver/rival: driver names (as in laps['Driver'])
    - candidate_pit_lap: lap when 'driver' would pit
    - pit_delta_est: estimated pit stop time cost (s) (incl. pit-lane)
    Returns:
    - dict with simulated relative RaceTime_s at end of race (lower is better) and estimated net gain.
    Approach:
    - When driver pits at candidate_pit_lap, we simulate that their RaceTime_s increases by pit_delta_est on that lap,
      and subsequent laps continue using actual lap times (we assume tyres after pit follow the average of next stint or average)
    - For rival, we assume they pit at their actual pit lap(s) from data. If they pit after driver, the model shows if undercut worked.
    NOTE: This is a heuristic simulation; for production we replace with Monte-Carlo.
    """
    # defensive checks
    if laps.empty:
        return {"error": "lap data empty"}

    # extract per-lap times for driver/rival
    d_laps = laps[laps['Driver'] == driver].set_index('LapNumber').sort_index()
    r_laps = laps[laps['Driver'] == rival].set_index('LapNumber').sort_index()

    max_lap = int(laps['LapNumber'].max())

    # build simulated cumulative times arrays for each lap
    sim_d = []
    sim_r = []
    d_cum = 0.0
    r_cum = 0.0

    # estimate driver's pit lap actual if exists in detected pit events; we override with candidate
    detected = detect_pit_stops_from_compound(laps)
    rival_pit_laps = detected[detected['Driver'] == rival]['pit_lap'].tolist()
    if len(rival_pit_laps) == 0:
        # guess rival pits at mid-race if none detected
        rival_pit_laps = [max_lap // 2]

    for lap in range(1, max_lap + 1):
        # driver's lap time
        if lap in d_laps.index:
            d_time = float(d_laps.loc[lap]['LapTime_s'])
        else:
            d_time = np.nan

        if lap in r_laps.index:
            r_time = float(r_laps.loc[lap]['LapTime_s'])
        else:
            r_time = np.nan

        # apply pit delta on driver if this is the candidate pit lap
        if lap == candidate_pit_lap:
            # driver loses pit delta plus lap time (pit lap often slower)
            if not np.isnan(d_time):
                d_time += pit_delta_est
            else:
                d_time = pit_delta_est

        # apply rival pit if lap in their pit laps (we add pit delta only on those laps)
        if lap in rival_pit_laps:
            if not np.isnan(r_time):
                r_time += pit_delta_est
            else:
                r_time = pit_delta_est

        # accumulate (if NaN treat as previous mean)
        if np.isnan(d_time):
            # fallback to mean lap time
            d_time = np.nanmean(d_laps['LapTime_s'])
        if np.isnan(r_time):
            r_time = np.nanmean(r_laps['LapTime_s'])

        d_cum += d_time
        r_cum += r_time
        sim_d.append(d_cum)
        sim_r.append(r_cum)

    # final gap (driver - rival); negative means driver ahead
    final_gap = sim_d[-1] - sim_r[-1]
    return {
        "driver": driver,
        "rival": rival,
        "candidate_pit_lap": candidate_pit_lap,
        "rival_pit_laps": rival_pit_laps,
        "final_gap_s": final_gap,
        "sim_driver_cum": sim_d,
        "sim_rival_cum": sim_r
    }

def stint_consistency(stints):
    """
    Compute per-stint statistics: mean, std, count, mean_delta_to_first
    """
    if stints.empty:
        return pd.DataFrame()

    # group per driver, stint
    stats = stints.groupby(['Driver', 'Stint'], as_index=False).agg(
        mean_lap_s=('LapTime_s', 'mean'),
        std_lap_s=('LapTime_s', 'std'),
        laps_in_stint=('LapNumber', 'count')
    )
    return stats

def sector_analysis(laps):
    """
    Prepare sector-level data: sector times are available in FastF1 as S1, S2, S3 Timedeltas (or Sector1Time etc).
    Convert to seconds and return tidy table.
    """
    if laps.empty:
        return pd.DataFrame()

    sector_cols = [c for c in laps.columns if 'Sector' in c or 's1' in c.lower() or 'S1' in c]
    # attempt common FastF1 names
    candidates = {}
    if 'Sector1Time' in laps.columns:
        candidates['S1'] = 'Sector1Time'
    if 'Sector2Time' in laps.columns:
        candidates['S2'] = 'Sector2Time'
    if 'Sector3Time' in laps.columns:
        candidates['S3'] = 'Sector3Time'

    # fallback to looking for 'S1' like columns
    for c in laps.columns:
        if c.lower().startswith('s') and 'sector' in c.lower():
            pass

    # build tidy df
    out_rows = []
    for _, row in laps.iterrows():
        driver = row.get('Driver')
        lap = row.get('LapNumber')
        for key, col in candidates.items():
            val = row.get(col)
            if pd.isnull(val):
                continue
            sec = val.total_seconds() if hasattr(val, 'total_seconds') else float(val)
            out_rows.append({'Driver': driver, 'LapNumber': lap, 'Sector': key, 'SectorTime_s': sec})
    if not out_rows:
        return pd.DataFrame()
    return pd.DataFrame(out_rows)

