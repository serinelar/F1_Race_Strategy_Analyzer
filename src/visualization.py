import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas as pd

def plot_lap_times(stints):
    """Lap time vs Lap (colored by tyre compound)."""
    fig = px.line(
        stints,
        x='LapNumber',
        y='LapTime_s',
        color='Compound',
        line_group='Driver',
        hover_data=['Driver', 'Stint'],
        title='Lap Time vs Lap (Colored by Compound)',
    )
    fig.update_layout(xaxis_title="Lap", yaxis_title="Lap Time (s)")
    return fig


def plot_pit_timeline(laps):
    """Pit stop Gantt chart."""
    pit_stops = laps[laps['PitOutTime'].notna()].copy()
    if pit_stops.empty:
        return None

    pit_stops['Start'] = pit_stops['PitInTime']
    pit_stops['Finish'] = pit_stops['PitOutTime']
    pit_stops['Task'] = pit_stops['Driver']
    pit_stops['Compound'] = pit_stops['Compound']

    fig = ff.create_gantt(
        pit_stops[['Task', 'Start', 'Finish', 'Compound']],
        index_col='Compound',
        show_colorbar=True,
        group_tasks=True,
        title="Pit Stop Timeline (per Driver)"
    )
    return fig


def plot_degradation_curve(degradation):
    """Average tyre degradation curve."""
    fig = px.line(
        degradation,
        x='LapIndex',
        y='LapTime_s',
        color='Compound',
        title='Tyre Degradation Curve'
    )
    fig.update_layout(xaxis_title="Laps in Stint", yaxis_title="Mean Lap Time (s)")
    return fig

def plot_gap_to_leader(gap_df):
    """
    gap_df: DataFrame returned by compute_gap_to_leader (contains LapNumber, Driver, GapToLeader_s)
    We'll plot lines per driver.
    """
    fig = go.Figure()
    # leader is driver with GapToLeader_s == 0 on each Lap
    leader = gap_df.loc[gap_df.groupby('LapNumber')['GapToLeader_s'].idxmin()]
    leader_line = leader.groupby('LapNumber')['LeaderTime_s'].first().reset_index()
    fig.add_trace(go.Scatter(x=leader_line['LapNumber'], y=leader_line['LeaderTime_s'],
                             mode='lines', name='Leader cumulative time'))
    
    fig = px.line(
        gap_df,
        x='LapNumber',
        y='GapToLeader_s',
        color='Driver',
        line_group='Driver',
        hover_data=['Driver', 'LapNumber'],
        title="Gap to Leader (seconds) vs Lap"
    )
    fig.update_yaxes(title="Gap to Leader (s)")
    fig.update_xaxes(title="Lap")
    return fig

def plot_stint_consistency(stats_df):
    """
    Bar chart of standard deviation per stint.
    """
    if stats_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No stint stats")
        return fig

    stats_df['Driver_Stint'] = stats_df['Driver'].astype(str) + "_S" + stats_df['Stint'].astype(str)
    fig = px.bar(
        stats_df,
        x='Driver_Stint',
        y='std_lap_s',
        hover_data=['mean_lap_s', 'laps_in_stint'],
        title="Stint Consistency (std of lap time)"
    )
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(title="Std Dev (s)")
    return fig

def plot_sector_violin(sector_df):
    """
    sector_df: tidy DataFrame with Driver, Sector, SectorTime_s
    """
    if sector_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No sector data")
        return fig

    fig = px.violin(
        sector_df,
        x='Sector',
        y='SectorTime_s',
        color='Sector',
        box=True,
        points='suspectedoutliers',
        title="Sector time distribution by sector"
    )
    fig.update_yaxes(title="Sector Time (s)")
    return fig

def plot_simulation(sim_dict, driver, rival):
    """ Plot cumulative race time simulation for driver and rival. """
    if not sim_dict or 'sim_driver_cum' not in sim_dict:
        fig = go.Figure()
        fig.update_layout(title="No simulation results")
        return fig

    lap_axis = list(range(1, len(sim_dict['sim_driver_cum']) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lap_axis, y=sim_dict['sim_driver_cum'],
                             mode='lines', name=driver))
    fig.add_trace(go.Scatter(x=lap_axis, y=sim_dict['sim_rival_cum'],
                             mode='lines', name=rival))
    fig.update_layout(
        title=f"Simulation: {driver} vs {rival} (final gap {sim_dict['final_gap_s']:.2f}s)",
        xaxis_title="Lap",
        yaxis_title="Cumulative Race Time (s)"
    )
    return fig


