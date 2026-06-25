"""
PlaceMux · Proctoring Hardening · Sample Data Generator
Generates realistic proctoring session data for models training.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

def generate_proctoring_data(n_samples: int = 2000) -> pd.DataFrame:
    """
    Generate synthetic but realistic proctoring session data.
    
    Features represent signals collected during online assessments:
    - Gaze / eye-tracking metrics
    - Keystroke dynamics
    - Mouse movement patterns
    - Tab/window switching events
    - Audio anomaly flags
    - Device environment signals
    """

    n_legit = int(n_samples * 0.72)   # ~72 % genuine
    n_cheat = n_samples - n_legit     # ~28 % cheating

    # ── Legitimate sessions ────────────────────────────────────────────
    legit = pd.DataFrame({
        # Gaze
        "gaze_off_screen_pct":       np.clip(np.random.beta(1.5, 12, n_legit) * 100, 0, 100),
        "avg_gaze_deviation_px":     np.random.normal(45, 15, n_legit).clip(0),
        "gaze_fixation_count":       np.random.normal(180, 40, n_legit).clip(20).astype(int),

        # Keystroke
        "keystroke_rhythm_score":    np.random.normal(0.82, 0.08, n_legit).clip(0, 1),
        "avg_keypress_interval_ms":  np.random.normal(210, 50, n_legit).clip(80),
        "copy_paste_count":          np.random.poisson(1.2, n_legit),
        "backspace_ratio":           np.random.beta(2, 8, n_legit),

        # Mouse
        "mouse_speed_variance":      np.random.normal(0.3, 0.1, n_legit).clip(0),
        "suspicious_click_pattern":  np.random.binomial(1, 0.04, n_legit),

        # Window / tab
        "tab_switch_count":          np.random.poisson(1.5, n_legit),
        "window_blur_duration_sec":  np.random.exponential(3, n_legit).clip(0),
        "fullscreen_exit_count":     np.random.poisson(0.5, n_legit),

        # Audio
        "audio_anomaly_count":       np.random.poisson(0.4, n_legit),
        "background_noise_level_db": np.random.normal(28, 8, n_legit).clip(10),

        # Device / environment
        "multiple_face_detected":    np.random.binomial(1, 0.02, n_legit),
        "face_absent_pct":           np.clip(np.random.beta(1, 15, n_legit) * 100, 0, 100),
        "phone_detected":            np.random.binomial(1, 0.01, n_legit),
        "lighting_score":            np.random.normal(0.78, 0.12, n_legit).clip(0, 1),
        "session_duration_min":      np.random.normal(52, 12, n_legit).clip(10, 120),
        "answer_edit_cycles":        np.random.poisson(3.5, n_legit),
        "label": 0,
    })

    # ── Cheating sessions ──────────────────────────────────────────────
    cheat = pd.DataFrame({
        "gaze_off_screen_pct":       np.clip(np.random.beta(4, 4, n_cheat) * 100, 15, 100),
        "avg_gaze_deviation_px":     np.random.normal(120, 40, n_cheat).clip(0),
        "gaze_fixation_count":       np.random.normal(90, 35, n_cheat).clip(5).astype(int),

        "keystroke_rhythm_score":    np.random.normal(0.45, 0.18, n_cheat).clip(0, 1),
        "avg_keypress_interval_ms":  np.where(
            np.random.random(n_cheat) < 0.4,
            np.random.normal(80, 20, n_cheat),    # copy-paste speed
            np.random.normal(350, 80, n_cheat)    # dictation lag
        ),
        "copy_paste_count":          np.random.poisson(8, n_cheat),
        "backspace_ratio":           np.random.beta(2, 4, n_cheat),

        "mouse_speed_variance":      np.random.normal(0.75, 0.2, n_cheat).clip(0),
        "suspicious_click_pattern":  np.random.binomial(1, 0.55, n_cheat),

        "tab_switch_count":          np.random.poisson(9, n_cheat),
        "window_blur_duration_sec":  np.random.exponential(18, n_cheat).clip(0),
        "fullscreen_exit_count":     np.random.poisson(4, n_cheat),

        "audio_anomaly_count":       np.random.poisson(4.5, n_cheat),
        "background_noise_level_db": np.random.normal(52, 14, n_cheat).clip(10),

        "multiple_face_detected":    np.random.binomial(1, 0.35, n_cheat),
        "face_absent_pct":           np.clip(np.random.beta(3, 4, n_cheat) * 100, 0, 100),
        "phone_detected":            np.random.binomial(1, 0.30, n_cheat),
        "lighting_score":            np.random.normal(0.52, 0.2, n_cheat).clip(0, 1),
        "session_duration_min":      np.random.normal(38, 18, n_cheat).clip(5, 120),
        "answer_edit_cycles":        np.random.poisson(12, n_cheat),
        "label": 1,
    })

    df = pd.concat([legit, cheat], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    df["session_id"] = [f"SES{str(i).zfill(5)}" for i in range(len(df))]

    # ── Realistic noise & class overlap (makes models comparison meaningful) ──
    rng = np.random.RandomState(99)
    noise_idx = rng.choice(len(df), size=int(len(df) * 0.08), replace=False)
    df.loc[noise_idx, "label"] = 1 - df.loc[noise_idx, "label"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(["label"])
    for col in numeric_cols:
        noise = rng.normal(0, df[col].std() * 0.15, size=len(df))
        df[col] = (df[col] + noise).clip(lower=0)
    n_borderline = int(n_samples * 0.12)
    border_idx = rng.choice(len(df), size=n_borderline, replace=False)
    for col in numeric_cols:
        midpoint = (df.loc[df["label"]==0, col].mean() + df.loc[df["label"]==1, col].mean()) / 2
        df.loc[border_idx, col] = (df.loc[border_idx, col] * 0.5 + midpoint * 0.5).clip(lower=0)
    return df


if __name__ == "__main__":
    df = generate_proctoring_data(2000)
    df.to_csv("proctoring_sessions.csv", index=False)
    print(f"Generated {len(df)} sessions  |  Cheating rate: {df['label'].mean():.1%}")
    print(df.describe().T[["mean","std","min","max"]].round(3))
