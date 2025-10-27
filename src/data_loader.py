import fastf1
import pandas as pd
import os

# --- Ensure cache directory exists ---
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Enable cache ---
fastf1.Cache.enable_cache(CACHE_DIR)

def load_session(year: int, gp_name: str, session_type: str = "R"):
    """
    Load F1 session data using FastF1.
    session_type: "R" for Race, "Q" for Qualifying, etc.
    """
    try:
        session = fastf1.get_session(year, gp_name, session_type)
        session.load()
        return session
    except Exception as e:
        print(f"❌ Failed to load session: {e}")
        return None
