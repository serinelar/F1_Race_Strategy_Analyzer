import streamlit as st
from data_loader import load_session
from strategy_analysis import *
from visualization import *
from strategy_recommender import recommend_optimal_strategy
from circuit_data import circuit_profiles

st.set_page_config(page_title="🏎️ F1 Race Strategy Analyzer", layout="wide")

st.title("🏎️ Formula 1 Race Strategy Analyzer (FastF1-Powered)")
st.markdown("Interactive, data-driven analysis of F1 races and tyre strategies.")

# Sidebar controls
year = st.sidebar.selectbox("Select Year", list(range(2018, 2024))[::-1])
gp_name = st.sidebar.text_input("Enter Grand Prix Name (e.g. 'Monza', 'Silverstone')", "Monza")
session_type = st.sidebar.selectbox("Session Type", ["R", "Q"])

# Load data only once
if "session_data" not in st.session_state:
    st.session_state.session_data = None

if st.sidebar.button("Load Data"):
    with st.spinner("Loading session data..."):
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
    cons_stats = stint_consistency(stints)
    st.plotly_chart(plot_stint_consistency(cons_stats), use_container_width=True)

    sector_df = sector_analysis(laps)
    st.plotly_chart(plot_sector_violin(sector_df), use_container_width=True)

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
        # Retrieve circuit profile
        circuit_info = circuit_profiles.get(gp_name, {"type": "balanced", "pit_loss": 20.0})
        st.write(f"📍 **Circuit profile:** {circuit_info['type']} | {circuit_info['notes']}")
        
        weather = st.radio("Weather Conditions", ["dry", "wet"], horizontal=True)
        pit_loss_input = st.slider("Estimated pit loss (s)", 15.0, 30.0, circuit_info["pit_loss"])
        
        best, all_strats = recommend_optimal_strategy(
            total_laps,
            degradation,
            pit_loss=pit_loss_input,
            circuit_type=circuit_info["type"],
            weather=weather
        )
        
        if best is not None:
            st.success(f"🏁 **Optimal strategy:** {best['Strategy']} — Total time: {best['TotalTime_s']:.1f} s")
            st.dataframe(all_strats)
        
        else:
            st.warning("No valid strategy could be generated for this configuration.")

