"""Driver Impairment Classification & Scoring Engine (Hybrid ML + Clinical Rule Matrix).

Combines trained gradient-boosted involuntary ocular models
(models/involuntary_model.joblib) with deterministic NHTSA/DRE clinical rules,
feeding calibrated probability streams into the BayesianEvidenceAccumulator.

Evaluates biological markers of alcohol intoxication and drowsiness:
  1. Static Image Analysis:
     - Ptosis (eyelid droop / low awake EAR)
     - Eyelid aperture asymmetry (|EAR_left - EAR_right|)
     - Cervical muscle slump / head tilt (pitch/roll hypotonia)
     - Jaw slackness (resting mouth aspect ratio)
     - Conjunctival redness (sclera vascular dilation / bloodshot index)
  2. Temporal Video Analysis:
     - Saccadic velocity degradation & pursuit fragmentation
     - Involuntary blink upstroke/downstroke slowing
     - Gaze-Evoked Nystagmus (GEN) & Lack of Convergence (LOC)
     - Gaze tunneling (reduced σ_yaw), PERCLOS & microsleep episodes
     - Real-time rolling-window ML probability fusion
"""
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

from pipeline.feature_extractor import FrameFeatures, is_monochrome_frame
from utils.landmarks import RIGHT_EYE_CONTOUR, LEFT_EYE_CONTOUR


class StarterLockout:
    ALLOW = False
    LOCK = True


class ThrottleState:
    NORMAL = 0
    LIMITED = 1


@dataclass
class ImpairmentAssessment:
    """Diagnostic outcome from the impairment classification engine."""
    risk_score: float = 0.0          # 0.0 to 100.0%
    classification: str = "SOBER"    # "SOBER", "MILD_IMPAIRMENT", "HIGH_RISK_INTOXICATED", "INSUFFICIENT_DATA"
    confidence: float = 0.0          # 0.0 to 1.0
    primary_indicators: List[str] = field(default_factory=list)
    marker_scores: Dict[str, float] = field(default_factory=dict)
    summary_text: str = ""
    involuntary_risk_score: float = 0.0
    voluntary_masking_score: float = 0.0
    masking_divergence: float = 0.0
    lockout_recommended: bool = False
    ml_probability: Optional[float] = None
    ml_model_active: bool = False
    involuntary_score: float = 0.0
    voluntary_score: float = 0.0
    total_risk_score: float = 0.0
    starter_lockout: bool = StarterLockout.ALLOW
    throttle_state: int = ThrottleState.NORMAL
    reason: str = ""


class ImpairmentClassifier:
    """Hybrid neuro-symbolic impairment classifier fusing NHTSA/DRE clinical rules with rolling-window ML inference."""

    def __init__(self,
                 sober_ear_baseline: float = 0.28,  # [PLACEHOLDER — pending calibration against real data]
                 sober_saccade_baseline: float = 80.0,  # [PLACEHOLDER — pending calibration against real data]
                 sober_blink_up_baseline: float = 2.5,  # [PLACEHOLDER — pending calibration against real data]
                 baseline_asymmetry: float = 0.0,  # [PLACEHOLDER — natural resting asymmetry offset]
                 model_path: Optional[str] = None,
                 fps: float = 30.0,
                 window_sec: float = 3.0,
                 max_occlusion_hold_frames: int = 15):
        """
        Args:
            sober_ear_baseline: Expected normal awake EAR for an alert driver.
            sober_saccade_baseline: Normal peak saccadic speed (deg/s).
            sober_blink_up_baseline: Normal eyelid reopening velocity (EAR/s).
            baseline_asymmetry: Driver's natural resting eyelid asymmetry (|ΔEAR| offset)
                subtracted from raw asymmetry to prevent false positive ptosis alerts.
            model_path: Optional file path to trained .joblib ML classifier artifact.
            fps: Expected stream framerate for physical-time ML window sizing (default 30.0).
            window_sec: Physical window duration in seconds for ML summary features (default 3.0s).
            max_occlusion_hold_frames: Max consecutive missing-face frames before clearing ML buffer.
        """
        self.sober_ear = sober_ear_baseline
        self.sober_saccade = sober_saccade_baseline
        self.sober_blink_up = sober_blink_up_baseline
        self.baseline_asymmetry = baseline_asymmetry
        self.ingress_impairment_detected: bool = False
        self.ingress_reason: str = ""
        self.pitch_baseline: float = 0.0
        self.roll_baseline: float = 0.0

        # Physical-time ML window configuration
        self.fps: float = float(max(fps, 1.0))
        self.window_sec: float = float(max(window_sec, 1.0))
        self._ml_buffer_len: int = max(30, int(round(self.fps * self.window_sec)))
        self._ml_min_infer_len: int = max(30, int(round(self.fps * 1.0)))
        self._max_occlusion_hold_frames: int = max(1, int(max_occlusion_hold_frames))
        self._consecutive_invalid_frames: int = 0

        # ML model attributes
        self.model_path = model_path
        self._ml_model = None
        self._ml_feature_names: List[str] = []
        self._ml_frame_buffer: Deque[np.ndarray] = deque(maxlen=self._ml_buffer_len)
        self._ml_feature_window = self._ml_frame_buffer
        self._ml_infer_stride: int = 5
        self._ml_eval_counter: int = 0
        self._last_ml_prob: Optional[float] = None

        if model_path is not None and os.path.exists(model_path):
            try:
                import joblib
                data = joblib.load(model_path)
                if isinstance(data, dict):
                    self._ml_model = data.get("model", None)
                    self._ml_feature_names = list(data.get("feature_names", []))
                else:
                    self._ml_model = data
                    self._ml_feature_names = []
                logger.info("[DMS Classifier] Loaded ML model from %s (%d features)", model_path, len(self._ml_feature_names))
            except Exception as e:
                logger.warning("[DMS Classifier] Failed to load ML model from %s: %s", model_path, e)

    def reset(self):
        """Reset temporal state and ML frame buffer."""
        self.ingress_impairment_detected = False
        self.ingress_reason = ""
        self._ml_frame_buffer.clear()
        self._ml_eval_counter = 0
        self._last_ml_prob = None
        self._consecutive_invalid_frames = 0

    def _predict_ml_probability(self, features: FrameFeatures) -> Optional[float]:
        """Predict real-time impairment probability over rolling physical window using trained ML model."""
        if self._ml_model is None:
            return None

        if not features.face_valid or features.imu_gated:
            return self._last_ml_prob

        try:
            from pipeline.training_buffer import TrainingBuffer
            numeric = [
                TrainingBuffer.clamp_numeric_feature(name, float(getattr(features, name, 0.0)))
                for name in TrainingBuffer.EXTENDED_NUMERIC_NAMES
            ]
            bools = [float(bool(getattr(features, name, False))) for name in TrainingBuffer.EXTENDED_BOOL_NAMES]
            vec = np.array(numeric + bools, dtype=np.float32)
        except Exception as e:
            logger.debug("[DMS Classifier] Vector extraction failed: %s", e)
            return self._last_ml_prob

        self._ml_frame_buffer.append(vec)
        self._ml_eval_counter += 1

        if len(self._ml_frame_buffer) >= self._ml_min_infer_len and (self._ml_eval_counter % self._ml_infer_stride == 0):
            try:
                from train_baseline import compute_window_summary_features
                from pipeline.training_buffer import TrainingBuffer

                win_3d = np.array(self._ml_frame_buffer, dtype=np.float32)[None, :, :]
                feat_2d, summary_names = compute_window_summary_features(
                    win_3d, TrainingBuffer.EXTENDED_FEATURE_NAMES, apply_subject_deltas=False
                )

                if self._ml_feature_names:
                    feat_map = {fn: idx for idx, fn in enumerate(summary_names)}
                    X_aligned = np.zeros((1, len(self._ml_feature_names)), dtype=np.float32)
                    for j, target_fn in enumerate(self._ml_feature_names):
                        if target_fn in feat_map:
                            X_aligned[0, j] = feat_2d[0, feat_map[target_fn]]
                else:
                    X_aligned = feat_2d

                if hasattr(self._ml_model, "predict_proba"):
                    probs = self._ml_model.predict_proba(X_aligned)
                    prob = float(probs[0, 1])
                elif hasattr(self._ml_model, "predict"):
                    prob = float(self._ml_model.predict(X_aligned)[0])
                else:
                    prob = None

                if prob is not None:
                    self._last_ml_prob = prob
            except Exception as e:
                logger.debug("[DMS Classifier] ML inference error: %s", e)

        return self._last_ml_prob

    def evaluate_static_image(self,
                              frame: np.ndarray,
                              landmarks: np.ndarray,
                              ear_left: float,
                              ear_right: float,
                              mar: float,
                              head_pitch: float,
                              head_roll: float,
                              speech_active: bool = False) -> ImpairmentAssessment:
        """Evaluate intoxication indicators from a single static facial image.

        Args:
            frame: BGR image array.
            landmarks: (478, 3) landmark coordinates in pixel space.
            ear_left: Left Eye Aspect Ratio.
            ear_right: Right Eye Aspect Ratio.
            mar: Mouth Aspect Ratio.
            head_pitch: Head pitch in degrees (negative = slumped down).
            head_roll: Head roll in degrees (tilt).

        Returns:
            ImpairmentAssessment dataclass.
        """
        marker_scores = {}
        indicators = []

        ear_avg = (ear_left + ear_right) / 2.0

        # ── 1. Ptosis / Eyelid Droop (Depressant CNS effect) ──
        # Awake sober eye: 0.26 - 0.35. Droopy drunk eye: 0.14 - 0.22. Closed: <0.12.
        if ear_avg < 0.12:  # [PLACEHOLDER — pending calibration against real data]
            # Eyes fully closed
            ptosis_score = 90.0
            indicators.append(f"Eyes nearly closed/shut (EAR={ear_avg:.2f})")
        elif ear_avg < 0.22:  # [PLACEHOLDER — pending calibration against real data]
            # Significant ptosis
            droop_pct = (0.26 - ear_avg) / (0.26 - 0.12)  # [PLACEHOLDER — pending calibration against real data]
            ptosis_score = float(np.clip(50.0 + droop_pct * 40.0, 50.0, 90.0))
            indicators.append(f"Pronounced eyelid droop/ptosis (EAR={ear_avg:.2f} vs norm >0.26)")
        elif ear_avg < 0.26:  # [PLACEHOLDER — pending calibration against real data]
            # Mild droop / drowsy
            ptosis_score = float(np.clip(20.0 + (0.26 - ear_avg) / 0.04 * 30.0, 20.0, 50.0))  # [PLACEHOLDER — pending calibration against real data]
            indicators.append(f"Mild eyelid relaxation (EAR={ear_avg:.2f})")
        else:
            ptosis_score = 0.0
        marker_scores["ptosis"] = ptosis_score

        # ── 2. Eyelid Asymmetry (Unequal levator palpebrae tone) ──
        raw_asymmetry = abs(ear_left - ear_right)
        asymmetry = max(0.0, raw_asymmetry - self.baseline_asymmetry)
        if asymmetry > 0.06:  # [PLACEHOLDER — pending calibration against real data]
            asym_score = 75.0
            indicators.append(
                f"Severe eyelid asymmetry (|ΔEAR|={raw_asymmetry:.2f}"
                + (f", excess={asymmetry:.2f})" if self.baseline_asymmetry > 0 else ")")
            )
        elif asymmetry > 0.035:  # [PLACEHOLDER — pending calibration against real data]
            asym_score = 40.0
            indicators.append(
                f"Moderate ocular asymmetry (|ΔEAR|={raw_asymmetry:.2f}"
                + (f", excess={asymmetry:.2f})" if self.baseline_asymmetry > 0 else ")")
            )
        else:
            asym_score = 0.0
        marker_scores["ocular_asymmetry"] = asym_score

        # ── 3. Head Slump / Loss of Cervical Tone ──
        # Pitch < -12 deg indicates forward drooping head. Roll > 12 deg indicates head list.
        head_slump_score = 0.0
        rel_pitch = head_pitch - self.pitch_baseline
        rel_roll = head_roll - self.roll_baseline
        if rel_pitch < -14.0:  # [PLACEHOLDER — pending calibration against real data]
            head_slump_score += min(60.0, abs(rel_pitch - (-14.0)) * 5.0)  # [PLACEHOLDER — pending calibration against real data]
            indicators.append(f"Cervical head slump (Pitch={head_pitch:.1f}°)")
        if abs(rel_roll) > 14.0:  # [PLACEHOLDER — pending calibration against real data]
            head_slump_score += min(40.0, (abs(rel_roll) - 14.0) * 4.0)  # [PLACEHOLDER — pending calibration against real data]
            indicators.append(f"Lateral head tilt/sway (Roll={head_roll:.1f}°)")
        marker_scores["head_slump"] = float(np.clip(head_slump_score, 0.0, 85.0))

        # ── 4. Jaw Slackness / Masseter Hypotonia ──
        # Normal resting closed mouth: MAR < 0.15. Resting slack jaw: 0.22 - 0.45.
        if speech_active:
            jaw_score = 0.0
        elif mar > 0.35:  # [PLACEHOLDER — pending calibration against real data]
            jaw_score = 70.0
            indicators.append(f"Slack jaw / mouth agape (MAR={mar:.2f})")
        elif mar > 0.22:  # [PLACEHOLDER — pending calibration against real data]
            jaw_score = 35.0
            indicators.append(f"Mild jaw relaxation (MAR={mar:.2f})")
        else:
            jaw_score = 0.0
        marker_scores["jaw_slackness"] = jaw_score

        # ── 5. Sclera Injection / Redness Index (Bloodshot Eyes) ──
        is_mono = is_monochrome_frame(frame)
        if not is_mono:
            sclera_redness = self._extract_sclera_redness(frame, landmarks)
            if sclera_redness > 1.35:  # [PLACEHOLDER — pending calibration against real data]
                redness_score = 75.0
                indicators.append(f"High sclera conjunctival injection/redness ({sclera_redness:.2f}x)")
            elif sclera_redness > 1.15:  # [PLACEHOLDER — pending calibration against real data]
                redness_score = 35.0
                indicators.append(f"Mild sclera redness ({sclera_redness:.2f}x)")
            else:
                redness_score = 0.0
            marker_scores["sclera_redness"] = redness_score
        else:
            marker_scores["sclera_redness"] = 0.0
            indicators.append("Monochrome/NIR active — sclera redness bypassed and weight redistributed")

        # ── Composite Risk Scoring (Weighted Static Formulation) ──
        # Sclera redness carries 5% corroborating weight in visible light.
        # Under monochrome / NIR, its 5% weight is dynamically redistributed across structural markers.
        if is_mono:
            weights = {
                "ptosis": 0.50 / 0.95,           # ~0.526
                "head_slump": 0.25 / 0.95,       # ~0.263
                "ocular_asymmetry": 0.10 / 0.95, # ~0.105
                "jaw_slackness": 0.10 / 0.95,    # ~0.105
            }
            base_risk = sum(marker_scores[k] * weights[k] for k in weights)
        else:
            weights = {
                "ptosis": 0.50,
                "head_slump": 0.25,
                "ocular_asymmetry": 0.10,
                "jaw_slackness": 0.10,
                "sclera_redness": 0.05,
            }
            base_risk = sum(marker_scores[k] * weights[k] for k in weights)

        # Co-occurrence compounding: if BOTH eyes are heavily drooping AND head is slumped,
        # depressant CNS inhibition is strongly indicated.
        if marker_scores["ptosis"] >= 60.0 and marker_scores["head_slump"] >= 25.0:  # [PLACEHOLDER — pending calibration against real data]
            base_risk += 15.0  # [PLACEHOLDER — pending calibration against real data]
            indicators.append("Compounding depressant indicators: Co-occurring ptosis and head slump")

        # Ingress alert: if driver biometrics at startup violated sober physiological bounds
        if self.ingress_impairment_detected:
            base_risk = max(base_risk, 55.0)
            reason_msg = f" ({self.ingress_reason})" if self.ingress_reason else ""
            indicators.insert(0, f"CRITICAL INGRESS ALERT: Driver biometrics at startup violated sober human physiological bounds{reason_msg}")

        total_risk = float(np.clip(base_risk, 0.0, 100.0))

        if total_risk >= 50.0:  # [PLACEHOLDER — pending calibration against real data]
            classification = "HIGH_RISK_INTOXICATED"
            confidence = float(np.clip(0.70 + (total_risk - 50.0) / 50.0 * 0.25, 0.70, 0.95))  # [PLACEHOLDER — pending calibration against real data]
        elif total_risk >= 25.0:  # [PLACEHOLDER — pending calibration against real data]
            classification = "MILD_IMPAIRMENT"
            confidence = 0.70
        else:
            classification = "SOBER"
            confidence = float(np.clip(0.80 + (25.0 - total_risk) / 25.0 * 0.15, 0.80, 0.95))  # [PLACEHOLDER — pending calibration against real data]

        summary = f"Risk Score: {total_risk:.1f}% | Classification: {classification}"
        lockout_rec = (classification == "HIGH_RISK_INTOXICATED")
        reason_str = "; ".join(indicators) if indicators else "Normal static baseline"
        return ImpairmentAssessment(
            risk_score=total_risk,
            classification=classification,
            confidence=confidence,
            primary_indicators=indicators,
            marker_scores=marker_scores,
            summary_text=summary,
            involuntary_risk_score=total_risk,
            lockout_recommended=lockout_rec,
            involuntary_score=total_risk,
            voluntary_score=0.0,
            total_risk_score=total_risk,
            starter_lockout=StarterLockout.LOCK if lockout_rec else StarterLockout.ALLOW,
            throttle_state=ThrottleState.LIMITED if classification in ("MILD_IMPAIRMENT", "HIGH_RISK_INTOXICATED") else ThrottleState.NORMAL,
            reason=reason_str,
        )

    def evaluate(self, *args, **kwargs) -> ImpairmentAssessment:
        """Convenience evaluation dispatcher supporting either FrameFeatures or static image arguments."""
        if args and isinstance(args[0], FrameFeatures):
            return self.evaluate_temporal_stream(args[0])
        if "features" in kwargs and isinstance(kwargs["features"], FrameFeatures):
            return self.evaluate_temporal_stream(kwargs["features"])
        return self.evaluate_static_image(*args, **kwargs)

    def evaluate_temporal_stream(self, features: FrameFeatures) -> ImpairmentAssessment:
        """Evaluate full temporal stream including saccadic, nystagmus & VOR dynamics.

        Args:
            features: FrameFeatures dataclass from FeatureExtractor.

        Returns:
            ImpairmentAssessment dataclass.
        """
        if not features.face_valid or features.imu_gated:
            # Preserve rolling ML window across brief IMU road shocks or short head-pose/occlusion drops;
            # only clear the buffer if the face has been invalid for > max_occlusion_hold_frames.
            if not features.face_valid:
                self._consecutive_invalid_frames += 1
                if self._consecutive_invalid_frames > self._max_occlusion_hold_frames:
                    self._ml_frame_buffer.clear()
                    self._last_ml_prob = None
            reason_msg = "Frame gated or face invalid — holding display only"
            return ImpairmentAssessment(
                risk_score=0.0,
                classification="INSUFFICIENT_DATA",
                confidence=0.0,
                primary_indicators=[reason_msg],
                marker_scores={},
                summary_text=f"Risk Score: 0.0% | Classification: INSUFFICIENT_DATA ({reason_msg})",
                involuntary_risk_score=0.0,
                voluntary_masking_score=0.0,
                involuntary_score=0.0,
                voluntary_score=0.0,
                total_risk_score=0.0,
                starter_lockout=StarterLockout.ALLOW,
                throttle_state=ThrottleState.NORMAL,
                reason=reason_msg,
            )

        self._consecutive_invalid_frames = 0

        marker_scores = {}
        indicators = []

        # 1. Gaze-Evoked Nystagmus (GEN — Romano et al. 2017)
        if features.gen_detected:
            marker_scores["gen"] = 90.0
            indicators.append(
                f"Gaze-Evoked Nystagmus (GEN): {features.gen_beat_count} beats, "
                f"centripetal drift {features.gen_slow_phase_vel:.1f}°/s"
            )
        else:
            marker_scores["gen"] = 0.0

        # 2. Smooth Pursuit Fragmentation Ratio
        if features.pursuit_fragmentation_ratio > 0.30:  # [PLACEHOLDER — pending calibration against real data]
            marker_scores["pursuit_fragmentation"] = 85.0
            indicators.append(
                f"Severe pursuit fragmentation ({features.pursuit_fragmentation_ratio*100:.0f}% catch-up saccades)"
            )
        elif features.pursuit_fragmentation_ratio > 0.15:  # [PLACEHOLDER — pending calibration against real data]
            marker_scores["pursuit_fragmentation"] = 45.0
            indicators.append(
                f"Elevated pursuit fragmentation ({features.pursuit_fragmentation_ratio*100:.0f}%)"
            )
        else:
            marker_scores["pursuit_fragmentation"] = 0.0

        # 3. Binocular Vergence & Lack of Convergence (LOC - DRE sign)
        if features.lack_of_convergence:
            marker_scores["lack_of_convergence"] = 80.0
            indicators.append(f"Lack of Convergence / divergent drift ({features.vergence_angle_deg:+.1f}°)")
        else:
            marker_scores["lack_of_convergence"] = 0.0

        # 4. Vestibulo-Ocular Reflex (VOR) Micro-Compensation Gain (PMC8997842)
        if features.vor_gain < 0.65:  # [PLACEHOLDER — pending calibration against real data]
            marker_scores["vor_depression"] = 75.0
            indicators.append(f"Depressed VOR micro-gain ({features.vor_gain:.2f} vs norm ~1.0)")
        elif features.vor_gain < 0.78:  # [PLACEHOLDER — pending calibration against real data]
            marker_scores["vor_depression"] = 35.0
            indicators.append(f"Mildly reduced VOR micro-gain ({features.vor_gain:.2f})")
        else:
            marker_scores["vor_depression"] = 0.0

        # 5. PERCLOS (Drowsiness/Prolonged closure)
        if features.perclos > 0.30:  # [PLACEHOLDER — pending calibration against real data]
            perclos_score = 90.0
            indicators.append(f"Severe PERCLOS ({features.perclos*100:.1f}% eyes closed)")
        elif features.perclos > 0.15:  # [PLACEHOLDER — pending calibration against real data]
            perclos_score = 50.0
            indicators.append(f"Elevated PERCLOS ({features.perclos*100:.1f}%)")
        else:
            perclos_score = 0.0
        marker_scores["perclos"] = perclos_score

        # 6. Blink Opening Velocity (Involuntary upstroke slowing)
        if features.blink_opening_velocity > 0:
            if features.blink_opening_velocity < 1.2:  # [PLACEHOLDER — pending calibration against real data]
                blink_vel_score = 80.0
                indicators.append(f"Sluggish blink upstroke ({features.blink_opening_velocity:.2f} EAR/s)")
            elif features.blink_opening_velocity < 1.8:  # [PLACEHOLDER — pending calibration against real data]
                blink_vel_score = 40.0
                indicators.append(f"Mildly slowed blink upstroke ({features.blink_opening_velocity:.2f} EAR/s)")
            else:
                blink_vel_score = 0.0
        else:
            blink_vel_score = 0.0
        marker_scores["blink_velocity"] = blink_vel_score

        # 7. Gaze Yaw Dispersion (Tunneling)
        tunnel_corroborated = (
            features.blink_opening_velocity == 0.0
            or features.perclos > 0.10
            or features.ear_avg < self.sober_ear * 0.88
            or features.gen_detected
            or features.lack_of_convergence
            or features.pursuit_fragmentation_ratio > 0.15
        )
        if features.gaze_yaw_dispersion > 0 and tunnel_corroborated:
            if features.gaze_yaw_dispersion < 0.03:  # [PLACEHOLDER — pending calibration against real data]
                tunnel_score = 75.0
                indicators.append(f"Gaze tunneling toward road center (σ_yaw={features.gaze_yaw_dispersion:.3f})")
            elif features.gaze_yaw_dispersion < 0.06:  # [PLACEHOLDER — pending calibration against real data]
                tunnel_score = 35.0
                indicators.append(f"Reduced scanning arc (σ_yaw={features.gaze_yaw_dispersion:.3f})")
            else:
                tunnel_score = 0.0
        else:
            tunnel_score = 0.0
        marker_scores["gaze_tunneling"] = tunnel_score

        # 8. Sustained Microsleep
        if features.microsleep_detected:
            indicators.append("Active micro-sleep episode detected")
            marker_scores["microsleep"] = 100.0
        else:
            marker_scores["microsleep"] = 0.0

        # 9. Low-Frequency Postural Sway (Section A Involuntary Micro-Tremor)
        # NOTE: Seated postural sway is an experimental proxy of Romberg standing posturography.
        if features.head_postural_sway > 1.8:
            marker_scores["postural_sway"] = 70.0
            indicators.append(f"Pronounced de-trended postural sway / micro-tremor (σ={features.head_postural_sway:.2f}°)")
        elif features.head_postural_sway > 1.2:
            marker_scores["postural_sway"] = 35.0
            indicators.append(f"Mild de-trended postural sway (σ={features.head_postural_sway:.2f}°)")
        else:
            marker_scores["postural_sway"] = 0.0

        # 10. Facial Flushing / Cheek Vasodilation (Section A Involuntary - RGB only)
        if features.flushing_valid and features.flushing_delta >= 0.20:
            marker_scores["facial_flushing"] = 70.0
            indicators.append(f"Facial vasodilation / cheek flushing (+{features.flushing_delta*100:.0f}% over baseline)")
        elif features.flushing_valid and features.flushing_delta >= 0.10:
            marker_scores["facial_flushing"] = 35.0
            indicators.append(f"Mild cheek flushing (+{features.flushing_delta*100:.0f}% over baseline)")
        else:
            marker_scores["facial_flushing"] = 0.0

        # Palpebral Eyelid Tremor (Section A Involuntary)
        if features.eyelid_tremor_hz >= 5.0:
            marker_scores["eyelid_tremor"] = 65.0
            indicators.append(f"Palpebral eyelid micro-tremor ({features.eyelid_tremor_hz:.1f} Hz flutter)")
        elif features.eyelid_tremor_hz >= 3.5:
            marker_scores["eyelid_tremor"] = 30.0
            indicators.append(f"Mild palpebral flutter ({features.eyelid_tremor_hz:.1f} Hz)")
        else:
            marker_scores["eyelid_tremor"] = 0.0

        # Ocular Eyeball Micro-Tremor / Flutter (Section A Involuntary)
        eye_tremor = getattr(features, "eyeball_tremor_hz", 0.0)
        if eye_tremor >= 5.0:
            marker_scores["eyeball_tremor"] = 65.0
            indicators.append(f"Ocular eyeball micro-tremor ({eye_tremor:.1f} Hz oscillation)")
        elif eye_tremor >= 3.5:
            marker_scores["eyeball_tremor"] = 30.0
            indicators.append(f"Mild ocular eyeball flutter ({eye_tremor:.1f} Hz)")
        else:
            marker_scores["eyeball_tremor"] = 0.0

        # ── Section B: Voluntary & Masking Behaviors ──
        # 11. Voluntary Eye-Widening Spikes (Levator/Frontalis compensation)
        if features.voluntary_eye_widening:
            marker_scores["voluntary_eye_widening"] = 75.0
            indicators.append("Voluntary eye-widening spike (transient ptosis counter-effort)")
        else:
            marker_scores["voluntary_eye_widening"] = 0.0

        # 12. Deliberate Blink Suppression
        if features.deliberate_blink_suppression:
            marker_scores["deliberate_blink_suppression"] = 70.0
            indicators.append("Deliberate blink suppression (prolonged inter-blink interval fighting droop)")
        else:
            marker_scores["deliberate_blink_suppression"] = 0.0

        # 13. Compensatory Stare Fixation
        if features.compensatory_stare_active:
            marker_scores["compensatory_stare"] = 75.0
            indicators.append(f"Compensatory stare fixation ({features.stare_fixation_duration_s:.1f}s rigid unbroken gaze, σ_gaze<0.8°)")
        elif features.stare_fixation_duration_s >= 6.0:
            marker_scores["compensatory_stare"] = 35.0
            indicators.append(f"Prolonged staring fixation ({features.stare_fixation_duration_s:.1f}s)")
        else:
            marker_scores["compensatory_stare"] = 0.0

        # 14. Rigid Head Stabilization
        suppress_rigid = (
            features.ear_avg >= self.sober_ear * 0.90
            and (features.blink_opening_velocity == 0.0 or features.blink_opening_velocity >= 1.50)
            and not features.gen_detected
            and not features.lack_of_convergence
            and features.perclos <= 0.10
        )
        if features.rigid_head_stabilization and not suppress_rigid:
            marker_scores["rigid_head_stabilization"] = 60.0
            indicators.append("Rigid head stabilization (unnatural neck tension suppressing sway)")
        else:
            marker_scores["rigid_head_stabilization"] = 0.0

        # 15. Frontalis Eyebrow Elevation (Section B Voluntary)
        if features.frontalis_compensation_active:
            marker_scores["frontalis_compensation"] = 65.0
            indicators.append(f"Frontalis eyebrow elevation (+{features.brow_elevation_delta*100:.0f}% brow lift fighting ptosis)")
        else:
            marker_scores["frontalis_compensation"] = 0.0

        # ── Anti-Masking Divergence Metric (M_mask) ──
        masking_divergence = features.masking_divergence_score
        marker_scores["masking_divergence"] = masking_divergence
        if features.masking_detected or masking_divergence >= 50.0:
            indicators.append(
                f"Anti-Masking Alert: High discordance between conscious effort and involuntary decay ({masking_divergence:.1f}% M_mask)"
            )

        # ── Lockout & Risk Weighting (Section A Anchored) ──
        # Primary involuntary weights (unfakeable physiological signals)
        # Note: Single-camera VOR is demoted to 0.0 weight for lockout decisions (Tier 3 experimental proxy)
        inv_weights = {
            "gen": 0.24,
            "pursuit_fragmentation": 0.22,
            "blink_velocity": 0.22,
            "vor_depression": 0.00,
            "lack_of_convergence": 0.12,
            "perclos": 0.10,
            "gaze_tunneling": 0.05,
            "microsleep": 0.05,
        }
        inv_risk = sum(marker_scores[k] * inv_weights[k] for k in inv_weights)

        # Autonomic corroboration bonus: only applied if base involuntary risk is already present (>= 20%)
        # Note on Seated Postural Sway: Without differential CAN-bus IMU subtraction of vehicle cabin
        # vibration and suspension harmonics (0.5-2.0 Hz), seated head sway cannot be distinguished from
        # rough road / chassis inputs. Therefore, it carries 0% authority in lockout and acts strictly as a corroborating bonus.
        if inv_risk >= 20.0:
            if marker_scores["postural_sway"] >= 35.0:
                inv_risk += 5.0
            if marker_scores["facial_flushing"] >= 35.0:
                inv_risk += 5.0
            if marker_scores.get("eyelid_tremor", 0.0) >= 30.0:
                inv_risk += 4.0
            if marker_scores.get("eyeball_tremor", 0.0) >= 30.0:
                inv_risk += 4.0
            if features.rppg_valid and 0.0 < features.rppg_hrv_rmssd_ms < 18.0:
                inv_risk += 3.0
                indicators.append(f"Depressed autonomic HRV (RMSSD={features.rppg_hrv_rmssd_ms:.1f}ms, BPM={features.rppg_pulse_bpm:.0f})")

        # Check if live driver currently exhibits healthy, alert open eyes with zero hard impairment signs
        live_alert_open_eyes = (
            features.ear_avg > 0.0
            and features.ear_avg >= self.sober_ear * 0.84
            and features.perclos < 0.12
            and not features.microsleep_detected
            and not features.gen_detected
            and not features.lack_of_convergence
            and features.pursuit_fragmentation_ratio <= 0.20
        )

        # Ingress alert: only floor at 55.0 if driver is NOT currently showing live alert open eyes
        # (Note: features.ear_avg == 0.0 in synthetic unit tests where ear_avg is omitted)
        if self.ingress_impairment_detected and (features.ear_avg == 0.0 or not live_alert_open_eyes):
            inv_risk = max(inv_risk, 55.0)
            reason_msg = f" ({self.ingress_reason})" if self.ingress_reason else ""
            indicators.insert(0, f"CRITICAL INGRESS ALERT: Driver biometrics at startup violated sober human physiological bounds{reason_msg}")

        inv_risk = float(np.clip(inv_risk, 0.0, 100.0))

        # Real-time ML model fusion (rolling physical window)
        ml_prob = self._predict_ml_probability(features)
        ml_prob_pct = None
        ml_active = False
        if ml_prob is not None:
            ml_prob_pct = round(ml_prob * 100.0, 1)
            ml_active = True
            marker_scores["ml_probability"] = ml_prob_pct
            if live_alert_open_eyes and inv_risk < 20.0:
                # Prevent YouTube domain shift from pushing an alert, wide-eyed sober webcam user into MILD/HIGH_RISK
                inv_risk = float(min(22.0, 0.85 * inv_risk + 0.15 * (ml_prob * 100.0)))
            else:
                ml_weight = 0.25 if (inv_risk < 15.0 and features.perclos < 0.10) else 0.40
                inv_risk = float(np.clip((1.0 - ml_weight) * inv_risk + ml_weight * (ml_prob * 100.0), 0.0, 100.0))
                if ml_prob >= 0.50:
                    indicators.append(f"Trained ML Classifier Probability: {ml_prob * 100:.1f}%")

        # Voluntary masking magnitude
        vol_keys = ["voluntary_eye_widening", "deliberate_blink_suppression", "compensatory_stare", "rigid_head_stabilization", "frontalis_compensation"]
        vol_scores = [marker_scores[k] for k in vol_keys if marker_scores[k] > 0]
        vol_masking = float(np.mean(vol_scores)) if vol_scores else 0.0

        # Lockout Decision: Anchored on Section A Involuntary evidence + Anti-masking divergence
        lockout_recommended = False
        if inv_risk >= 50.0:
            classification = "HIGH_RISK_INTOXICATED"
            lockout_recommended = True
            total_risk = inv_risk
        elif features.masking_detected and inv_risk >= 40.0:
            classification = "HIGH_RISK_INTOXICATED"
            lockout_recommended = True
            total_risk = max(inv_risk, masking_divergence)
        elif inv_risk >= 25.0 or (vol_masking >= 50.0 and inv_risk >= 22.0 and features.perclos > 0.08):
            classification = "MILD_IMPAIRMENT"
            total_risk = max(inv_risk, 25.0)
        else:
            classification = "SOBER"
            total_risk = inv_risk

        total_risk = float(np.clip(total_risk, 0.0, 100.0))

        reason_str = "; ".join(indicators) if indicators else "No impairment indicators"
        return ImpairmentAssessment(
            risk_score=total_risk,
            classification=classification,
            confidence=0.88,  # [PLACEHOLDER — pending calibration against real data]
            primary_indicators=indicators,
            marker_scores=marker_scores,
            summary_text=f"Temporal Risk: {total_risk:.1f}% | {classification}" + (" [LOCKOUT TRIGGERED]" if lockout_recommended else ""),
            involuntary_risk_score=inv_risk,
            voluntary_masking_score=vol_masking,
            masking_divergence=masking_divergence,
            lockout_recommended=lockout_recommended,
            ml_probability=ml_prob_pct,
            ml_model_active=ml_active,
            involuntary_score=inv_risk,
            voluntary_score=vol_masking,
            total_risk_score=total_risk,
            starter_lockout=StarterLockout.LOCK if lockout_recommended else StarterLockout.ALLOW,
            throttle_state=ThrottleState.LIMITED if (lockout_recommended or classification in ("MILD_IMPAIRMENT", "HIGH_RISK_INTOXICATED")) else ThrottleState.NORMAL,
            reason=reason_str,
        )

    @staticmethod
    def _extract_sclera_redness(frame: np.ndarray, landmarks: np.ndarray) -> float:
        """Measure redness ratio in the sclera (white of eye) regions.

        Uses cropped eye bounding box fast path (<0.05ms) instead of full-frame mask.

        Returns:
            Redness ratio: R / ((G + B) / 2 + 1e-5). Values > 1.2 indicate bloodshot eyes.
            Returns 1.0 (neutral) if image is monochrome / NIR.
        """
        if is_monochrome_frame(frame) or landmarks is None or len(landmarks) < 468:
            return 1.0

        try:
            h, w = frame.shape[:2]
            if h <= 0 or w <= 0:
                return 1.0
            eye_indices = RIGHT_EYE_CONTOUR + LEFT_EYE_CONTOUR
            pts = landmarks[eye_indices, :2]
            if np.isnan(pts).any() or np.isinf(pts).any():
                return 1.0
            pts_int = pts.astype(np.int32)
            x_min = max(0, int(np.min(pts_int[:, 0])) - 5)
            x_max = min(w, int(np.max(pts_int[:, 0])) + 5)
            y_min = max(0, int(np.min(pts_int[:, 1])) - 5)
            y_max = min(h, int(np.max(pts_int[:, 1])) + 5)

            if x_max <= x_min or y_max <= y_min:
                return 1.0

            crop = frame[y_min:y_max, x_min:x_max]
            if crop.size == 0 or crop.ndim != 3 or crop.shape[2] != 3:
                return 1.0

            mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)

            r_pts = (pts_int[:len(RIGHT_EYE_CONTOUR)] - [x_min, y_min])
            l_pts = (pts_int[len(RIGHT_EYE_CONTOUR):] - [x_min, y_min])

            cv2.fillPoly(mask, [r_pts], 255)
            cv2.fillPoly(mask, [l_pts], 255)

            # Exclude dark pupil/iris pixels (luminance < 70) to isolate white sclera
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sclera_mask = (mask == 255) & (gray > 70)  # [PLACEHOLDER — pending calibration against real data]

            if not np.any(sclera_mask):
                return 1.0

            b = crop[:, :, 0][sclera_mask].astype(np.float64)
            g = crop[:, :, 1][sclera_mask].astype(np.float64)
            r = crop[:, :, 2][sclera_mask].astype(np.float64)

            mean_r = np.mean(r)
            mean_gb = (np.mean(g) + np.mean(b)) / 2.0

            return float(mean_r / (mean_gb + 1e-5))
        except Exception as exc:
            logger.debug("Failed to extract sclera redness: %s", exc, exc_info=True)
            return 1.0
