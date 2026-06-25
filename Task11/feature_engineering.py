"""
PlaceMux · Proctoring Hardening · Feature Engineering
Derives higher-level risk signals from raw session telemetry.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler


FEATURE_COLS = [
    "gaze_off_screen_pct",
    "avg_gaze_deviation_px",
    "gaze_fixation_count",
    "keystroke_rhythm_score",
    "avg_keypress_interval_ms",
    "copy_paste_count",
    "backspace_ratio",
    "mouse_speed_variance",
    "suspicious_click_pattern",
    "tab_switch_count",
    "window_blur_duration_sec",
    "fullscreen_exit_count",
    "audio_anomaly_count",
    "background_noise_level_db",
    "multiple_face_detected",
    "face_absent_pct",
    "phone_detected",
    "lighting_score",
    "session_duration_min",
    "answer_edit_cycles",
]


class ProctoringFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Derives composite risk scores from raw proctoring telemetry.
    All new features are interpretable and directly mappable to human-readable
    explanations — no silent black-box aggregations.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # ── Composite risk scores (each maps to a plain-English signal) ──
        df["gaze_risk_score"] = (
            df["gaze_off_screen_pct"] / 100 * 0.5 +
            (df["avg_gaze_deviation_px"] / 200).clip(0, 1) * 0.3 +
            (1 - (df["gaze_fixation_count"] / 300).clip(0, 1)) * 0.2
        )

        df["keystroke_risk_score"] = (
            (1 - df["keystroke_rhythm_score"]) * 0.4 +
            (df["copy_paste_count"] / 20).clip(0, 1) * 0.4 +
            df["backspace_ratio"] * 0.2
        )

        df["window_risk_score"] = (
            (df["tab_switch_count"] / 20).clip(0, 1) * 0.4 +
            (df["window_blur_duration_sec"] / 60).clip(0, 1) * 0.35 +
            (df["fullscreen_exit_count"] / 10).clip(0, 1) * 0.25
        )

        df["face_risk_score"] = (
            df["multiple_face_detected"] * 0.4 +
            df["face_absent_pct"] / 100 * 0.35 +
            df["phone_detected"] * 0.25
        )

        df["audio_risk_score"] = (
            (df["audio_anomaly_count"] / 10).clip(0, 1) * 0.6 +
            ((df["background_noise_level_db"] - 30) / 40).clip(0, 1) * 0.4
        )

        # ── Aggregate overall risk ──────────────────────────────────────
        df["overall_risk_score"] = (
            df["gaze_risk_score"] * 0.20 +
            df["keystroke_risk_score"] * 0.25 +
            df["window_risk_score"] * 0.25 +
            df["face_risk_score"] * 0.20 +
            df["audio_risk_score"] * 0.10
        )

        # ── Interaction features ────────────────────────────────────────
        df["high_copy_low_rhythm"] = (
            (df["copy_paste_count"] > 5) & (df["keystroke_rhythm_score"] < 0.5)
        ).astype(int)

        df["face_absent_and_noise"] = (
            (df["face_absent_pct"] > 20) & (df["audio_anomaly_count"] > 2)
        ).astype(int)

        df["tab_and_window_flag"] = (
            (df["tab_switch_count"] > 5) & (df["window_blur_duration_sec"] > 10)
        ).astype(int)

        return df

    def explain(self, row: pd.Series) -> dict:
        """Return plain-English explanation for a single session prediction."""
        reasons = []

        if row.get("gaze_off_screen_pct", 0) > 25:
            reasons.append(f"Gaze off-screen {row['gaze_off_screen_pct']:.1f}% of session (threshold: 25%)")

        if row.get("copy_paste_count", 0) > 5:
            reasons.append(f"Excessive copy-paste: {int(row['copy_paste_count'])} events (threshold: 5)")

        if row.get("tab_switch_count", 0) > 5:
            reasons.append(f"Frequent tab switching: {int(row['tab_switch_count'])} times (threshold: 5)")

        if row.get("window_blur_duration_sec", 0) > 10:
            reasons.append(f"Window unfocused for {row['window_blur_duration_sec']:.0f}s (threshold: 10s)")

        if row.get("multiple_face_detected", 0) == 1:
            reasons.append("Multiple faces detected in camera frame")

        if row.get("phone_detected", 0) == 1:
            reasons.append("Mobile phone detected in camera frame")

        if row.get("face_absent_pct", 0) > 20:
            reasons.append(f"Candidate absent from frame {row['face_absent_pct']:.1f}% of session")

        if row.get("audio_anomaly_count", 0) > 3:
            reasons.append(f"Audio anomalies detected: {int(row['audio_anomaly_count'])} events")

        if row.get("keystroke_rhythm_score", 1) < 0.5:
            reasons.append(f"Abnormal keystroke rhythm (score: {row['keystroke_rhythm_score']:.2f})")

        if not reasons:
            reasons.append("No individual high-risk signals — flagged by combined pattern of weak signals")

        return {
            "risk_level": "HIGH" if row.get("overall_risk_score", 0) > 0.5 else "MEDIUM" if row.get("overall_risk_score", 0) > 0.3 else "LOW",
            "overall_risk_score": float(row.get("overall_risk_score", 0)),
            "reasons": reasons,
        }


def get_all_feature_cols(df: pd.DataFrame) -> list:
    """Return all feature columns (raw + engineered) excluding ID/label."""
    exclude = {"session_id", "label"}
    return [c for c in df.columns if c not in exclude]
