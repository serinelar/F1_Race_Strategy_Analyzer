# app.py
import streamlit as st
import fastf1
import pandas as pd

from data_loader import load_session
from strategy_analysis import *
from visualization import *
from strategy_recommender import recommend_optimal_strategy
from undercut_simulator import simulate_undercut_vs_overcut, plot_undercut_simulation
from insights import lap_time_insights

st.set_page_config(page_title="🏎️ F1 Race Strategy Analyzer", layout="wide")

st.title("🏎️ Formula 1 Race Strategy Analyzer (FastF1-Powered)")
st.markdown("Interactive, data-driven analysis of F1 races and tyre strategies.")

# Sidebar controls
year = st.sidebar.selectbox("Select Year", list(range(2018, 2024))[::-1])

# Fetch circuit names dynamically
@st.cache_data(show_spinner=False)
def get_circuit_names(year):
    try:
        schedule = fastf1.get_event_schedule(year)
        return schedule["EventName"].unique().tolist()
    except Exception as e:
        print(f"⚠️ Could not load event schedule for {year}: {e}")
        # Fallback to a few common names
        return ["Monza", "Silverstone", "Spa", "Bahrain", "Melbourne", "Suzuka"]

circuit_names = get_circuit_names(year)
gp_name = st.sidebar.selectbox("Select Grand Prix", circuit_names, index=0)

@st.cache_data(show_spinner=False)
def get_circuit_profile(_session):
    """
    Dynamically infer circuit characteristics from FastF1 metadata.
    """
    try:
        event = _session.event
        circuit_name = event['EventName']
        location = event['Location']
        country = event['Country']
        event_format = event['EventFormat']
        round_number = event['RoundNumber']

        # Try to extract circuit length and other info
        track_length_km = getattr(event, 'OfficialLength', None)
        if track_length_km is None:
            # heuristic fallback
            if "Monaco" in circuit_name:
                track_length_km = 3.34
            elif "Spa" in circuit_name:
                track_length_km = 7.00
            else:
                track_length_km = 5.0

        # Heuristic circuit classification
        lower = circuit_name.lower()
        if any(k in lower for k in ["monaco", "hungaroring", "singapore", "miami", "baku", "vegas", "jeddah"]):
            circuit_type = "street / high-downforce"
            pit_loss = 22.5
        elif any(k in lower for k in ["monza", "spa", "silverstone", "bahrain", "mexico", "cota"]):
            circuit_type = "low-downforce / power-sensitive"
            pit_loss = 18.5
        else:
            circuit_type = "balanced / technical"
            pit_loss = 20.0

        notes = (
            f"{circuit_name} ({country}) — Round {round_number} "
            f"| Track length ≈ {track_length_km:.2f} km "
            f"| Type: {circuit_type}"
        )

        return {
            "name": circuit_name,
            "type": circuit_type,
            "pit_loss": pit_loss,
            "length_km": track_length_km,
            "notes": notes,
        }

    except Exception as e:
        print(f"⚠️ Could not infer circuit profile: {e}")
        return {
            "name": "Unknown",
            "type": "balanced",
            "pit_loss": 20.0,
            "length_km": 5.0,
            "notes": "Default profile applied.",
        }

session_type = st.sidebar.selectbox("Session Type", ["R", "Q"])

# Load data only once
if "session_data" not in st.session_state:
    st.session_state.session_data = None

if st.sidebar.button("Load Data"):
    with st.spinner(f"Loading {year} {gp_name} {session_type} session..."):
        session = load_session(year, gp_name, session_type)
        if session is None:
            st.error("❌ Could not load session. Check GP name or internet connection.")
        else:
            laps = session.laps.pick_quicklaps()
            stints = compute_stint_data(laps)
            degradation = average_degradation(stints)
            st.session_state.session_data = (session, laps, stints, degradation)

# If data is loaded
if st.session_state.session_data:
    session, laps, stints, degradation = st.session_state.session_data

    # Main visuals
    st.subheader("📈 Lap Time Evolution")
    st.plotly_chart(plot_lap_times(stints), use_container_width=True)

    st.subheader("🧱 Pit Stop Timeline")
    pit_chart = plot_pit_timeline(laps)
    if pit_chart:
        st.plotly_chart(pit_chart, use_container_width=True)
    else:
        st.info("No pit data available for this session.")

    st.subheader("📊 Tyre Degradation Curve")
    st.plotly_chart(plot_degradation_curve(degradation), use_container_width=True)

    st.subheader("🔍 Auto insights")
    for insight in lap_time_insights(stints):
        st.write(f"- {insight}")

    # === Sidebar Tools ===
    st.sidebar.markdown("---")
    st.sidebar.header("Analysis tools")

    drivers_list = st.sidebar.multiselect("Select drivers to highlight", sorted(laps['Driver'].unique().tolist()), max_selections=2)

    st.sidebar.subheader("Undercut/Overcut simulator")
    sim_driver = st.sidebar.selectbox("Simulate: driver", laps['Driver'].unique())
    sim_rival = st.sidebar.selectbox("Against: rival", [d for d in laps['Driver'].unique() if d != sim_driver])
    candidate_pit = st.sidebar.number_input("Candidate pit lap for driver", min_value=1, max_value=int(laps['LapNumber'].max()), value=int(laps['LapNumber'].max()//2))
    pit_delta = st.sidebar.slider("Estimated pit delta (s)", 5.0, 40.0, 18.0)

    # === Analysis ===
    st.subheader("📈 Gap to leader")
    gap_df = compute_gap_to_leader(stints)
    st.plotly_chart(plot_gap_to_leader(gap_df), use_container_width=True)

    st.subheader("🔁 Undercut / Overcut simulation")
    sim_result = simulate_undercut_overtake(stints, sim_driver, sim_rival, candidate_pit, pit_delta_est=pit_delta)
    st.write(f"Simulated final gap (driver - rival) = {sim_result['final_gap_s']:.2f} s (negative means driver ahead)")
    st.plotly_chart(plot_simulation(sim_result, sim_driver, sim_rival), use_container_width=True)

    st.subheader("⚖️ Consistency & Sector analysis")
    st.plotly_chart(plot_stint_consistency(stint_consistency(stints)), use_container_width=True)
    st.plotly_chart(plot_sector_violin(sector_analysis(laps)), use_container_width=True)

    st.subheader("📡 Telemetry overlay")
    driver_choice = st.selectbox("Choose driver for telemetry", laps['Driver'].unique())
    try:
        tel = session.laps.pick_drivers(driver_choice).get_telemetry().add_distance()
        fig_tel = px.line(tel, x='Distance', y='Speed', title=f"Telemetry: Speed vs Distance ({driver_choice})")
        st.plotly_chart(fig_tel, use_container_width=True)
    except Exception as e:
        st.info("No telemetry available for selected driver/session.")
    
    # === Strategy Recommendation ===
    st.subheader("🧠 Recommended Race Strategy")
    
    total_laps = int(laps["LapNumber"].max())
    
    if degradation.empty:
        st.info("Degradation data unavailable to compute strategy recommendations.")
    
    else:
        circuit_info = get_circuit_profile(session)
        st.write(f"📍 **Circuit profile:** {circuit_info['type']} — {circuit_info['notes']}")

        weather = st.radio("Weather Conditions", ["dry", "wet"], horizontal=True)

        weather_forecast = st.radio("Weather Forecast", ["Dry", "Light Rain", "Heavy Rain", "Variable"], horizontal=True)
        
        # Apply bias factor
        weather_bias = {
            "Dry": 1.0,
            "Light Rain": 0.9,
            "Heavy Rain": 0.75,
            "Variable": 0.85
        }[weather_forecast]

        pit_loss_input = st.slider("Estimated pit loss (s)", 15.0, 30.0, circuit_info["pit_loss"])
        qual_pos = st.slider("Qualifying Position", 1, 20, 5)
        
        best, all_strats = recommend_optimal_strategy(
            total_laps,
            degradation,
            pit_loss=pit_loss_input,
            circuit_type=circuit_info["type"],
            weather=weather_forecast.lower(),
            qualifying_position=qual_pos,
            weather_bias=weather_bias,
            )
        
        if best is not None:
            st.success(f"🏁 **Optimal strategy:** {best['Strategy']} — Total time: {best['TotalTime_s']:.1f} s")
            st.dataframe(all_strats)
        
        else:
            st.warning("No valid strategy could be generated for this configuration.")

    
    st.markdown("---")
    st.header("⏱️ Undercut vs Overcut Payoff Simulation")
    
    compound_a = st.selectbox("Tyre for Driver A (Undercut)", degradation["Compound"].unique())
    compound_b = st.selectbox("Tyre for Driver B (Overcut)", degradation["Compound"].unique())
    undercut_lap = st.slider("Undercut Lap", 5, total_laps - 5, 15)
    pit_loss_sim = st.number_input("Pit Loss (s)", min_value=15.0, max_value=30.0, value=20.0)
    
    if st.button("Simulate Undercut/Overcut"):
        with st.spinner("Simulating..."):
            payoff_df, payoff_summary = simulate_undercut_vs_overcut(
                degradation, compound_a, compound_b, undercut_lap, total_laps, pit_loss_sim
            )
            fig = plot_undercut_simulation(payoff_df, payoff_summary)
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"✅ Optimal undercut lap: **{payoff_summary['BestLap']}** | Gap: **{payoff_summary['MinGap(s)']:.2f}s**")

