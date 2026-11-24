"""Analysis-mode integration: Layer 1 -> project C inverse DSP -> manifest.

Runs Layer 1 -> C inverse DSP -> Layer 2; optional direct Layer 2 diagnostic.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import wave


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILTER = ROOT / "Filters" / ("filter.exe" if os.name == "nt" else "filter")
SAMPLE_RATE = 48000
FILTER_TIMEOUT_SECONDS = 600


class AnalysisError(RuntimeError):
    pass


def wav_info(path: Path) -> dict:
    if not path.is_file():
        raise AnalysisError(f"WAV does not exist: {path}")
    try:
        with wave.open(str(path), "rb") as wav:
            rate, channels, width, frames, compression = (
                wav.getframerate(), wav.getnchannels(), wav.getsampwidth(),
                wav.getnframes(), wav.getcomptype(),
            )
    except (wave.Error, EOFError, OSError) as exc:
        raise AnalysisError(f"Invalid or unsupported WAV {path}: {exc}") from exc
    if (rate, channels, width, compression) != (SAMPLE_RATE, 1, 2, "NONE"):
        raise AnalysisError(
            f"{path}: C filters require mono 48,000-Hz 16-bit PCM WAV; "
            f"got {channels} channel(s), {rate} Hz, {width * 8}-bit, {compression}"
        )
    if frames < 1:
        raise AnalysisError(f"Empty WAV: {path}")
    return {"sample_rate_hz": rate, "channels": channels, "bits_per_sample": width * 8,
            "frames": frames, "duration_seconds": round(frames / rate, 6)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_spec(spec: object) -> tuple[dict, list[str]]:
    """Translate Layer 1's band-pass TOTAL order to C's PER-STAGE order."""
    if not isinstance(spec, dict):
        raise AnalysisError("Layer 1 result has no 'spec' object")
    kind = spec.get("type")
    family = spec.get("family", "butter")
    order = spec.get("order")
    cutoff = spec.get("cutoff")
    if kind not in ("lowpass", "highpass", "bandpass"):
        raise AnalysisError(f"Unsupported filter type: {kind!r}")
    if family != "butter":
        raise AnalysisError(f"C executable only supports 'butter', got {family!r}")
    if type(order) is not int:
        raise AnalysisError("Filter order must be an integer")
    allowable = (2, 4, 8) if kind == "bandpass" else (1, 2, 4)
    if order not in allowable:
        raise AnalysisError(f"Unsupported {kind} total order {order}; expected {allowable}")
    def valid_frequency(v: object) -> bool:
        return type(v) in (float, int) and math.isfinite(v) and 0 < v < SAMPLE_RATE / 2
    if kind == "bandpass":
        if not isinstance(cutoff, (list, tuple)) or len(cutoff) != 2:
            raise AnalysisError("Band-pass cutoff must be [lower_hz, upper_hz]")
        lo, hi = cutoff
        if not (valid_frequency(lo) and valid_frequency(hi) and lo < hi):
            raise AnalysisError(f"Invalid band-pass cutoff: {cutoff!r}")
        hp, lp = float(lo), float(hi)
        c_order, choice = order // 2, 6
    else:
        if not valid_frequency(cutoff):
            raise AnalysisError(f"Invalid {kind} cutoff: {cutoff!r}")
        # C CLI demands both frequencies be positive even for one-sided filtering.
        hp = float(cutoff) if kind == "highpass" else 1000.0
        lp = float(cutoff) if kind == "lowpass" else 1000.0
        c_order, choice = order, (5 if kind == "highpass" else 4)
    normalized = {"type": kind, "family": family, "order": order,
                  "cutoff": list(map(float, cutoff)) if kind == "bandpass" else float(cutoff)}
    args = [kind, "butter", str(c_order), f"{hp:.12g}", f"{lp:.12g}", str(choice)]
    return normalized, args



def lf_events_from_windows(windows: list[dict], labels: list[str]) -> list[dict]:
    """Combine adjacent, positive 5-s LF windows into per-label events.

    A negative window or time gap ends an event. Overlapping labels are kept
    independent; the original window-level predictions remain unchanged.
    """
    events = []
    for label in labels:
        active = None
        for win in windows:
            try:
                start = float(win["start_seconds"])
                end = float(win["end_seconds"])
                score = float(win["scores"][label])
                positive = win["detected"][label]
            except (KeyError, ValueError, TypeError) as exc:
                raise AnalysisError(f"LF prediction missing valid timing/score for {label}") from exc
            if (not math.isfinite(start) or not math.isfinite(end) or end <= start
                    or not math.isfinite(score) or not 0 <= score <= 1
                    or type(positive) is not bool):
                raise AnalysisError(f"LF prediction has invalid time/score/detection for {label}")
            contiguous = (active is not None and
                          math.isclose(start, active["end_seconds"], abs_tol=1e-6))
            if active is not None and (not positive or not contiguous):
                events.append({
                    "type": label,
                    "start_seconds": active["start_seconds"],
                    "end_seconds": active["end_seconds"],
                    "mean_confidence": round(active["score_sum"] / active["window_count"], 6),
                    "max_confidence": round(active["max_score"], 6),
                    "window_count": active["window_count"],
                })
                active = None
            if positive:
                if active is None:
                    active = {"start_seconds": start, "end_seconds": end,
                              "score_sum": score, "max_score": score, "window_count": 1}
                else:
                    active["end_seconds"] = end
                    active["score_sum"] += score
                    active["max_score"] = max(active["max_score"], score)
                    active["window_count"] += 1
        if active is not None:
            events.append({
                "type": label,
                "start_seconds": active["start_seconds"],
                "end_seconds": active["end_seconds"],
                "mean_confidence": round(active["score_sum"] / active["window_count"], 6),
                "max_confidence": round(active["max_score"], 6),
                "window_count": active["window_count"],
            })
    events.sort(key=lambda item: (item["start_seconds"], item["type"]))
    return events


def run_analysis(input_wav: str | Path, model_path: str | Path,
                 output_dir: str | Path, filter_binary: str | Path = DEFAULT_FILTER,
                 *, timeout: int = FILTER_TIMEOUT_SECONDS,
                 layer2_model: str | Path | None = None,
                 skip_inverse: bool = False,
                 threshold: float | None = None,
                 lf_model: str | Path | None = None,
                 lf_threshold: float = 0.5,
                 lf_cutoff: float | None = None,
                 lf_script: str | Path | None = None) -> Path:
    source = Path(input_wav).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    binary = Path(filter_binary).expanduser().resolve()
    l2model = Path(layer2_model).expanduser().resolve() if layer2_model is not None else None
    lfmodel = Path(lf_model).expanduser().resolve() if lf_model is not None else None
    lfscript = (Path(lf_script).expanduser().resolve() if lf_script is not None
                else ROOT / "LF_experiment" / "lf_experiment.py")
    if lf_cutoff is not None:
        if not math.isfinite(lf_cutoff) or not 0 < lf_cutoff < SAMPLE_RATE / 2:
            raise AnalysisError("--lf-cutoff must be finite and between 0 and 24000 Hz")
        if lfmodel is None:
            raise AnalysisError("--lf-cutoff requires an LF model")
    if lfmodel is not None and not lfmodel.is_file():
        raise AnalysisError(f"LF model not found: {lfmodel}")
    if lfmodel is not None and not lfscript.is_file():
        raise AnalysisError(f"LF prediction script not found: {lfscript}")
    if lfmodel is not None and not 0 < lf_threshold < 1:
        raise AnalysisError("--lf-threshold must be strictly between 0 and 1")
    if lfmodel is not None and skip_inverse:
        raise AnalysisError("--lf-model requires reconstruction; --skip-inverse is a direct Layer 2 diagnostic")
    if lfmodel is not None and not binary.is_file():
        raise AnalysisError(f"C filter executable not found: {binary}")
    if l2model is not None and not l2model.is_file():
        raise AnalysisError(f"Layer 2 model not found: {l2model}")
    if skip_inverse and l2model is None:
        raise AnalysisError("--skip-inverse requires --layer2-model")
    if not skip_inverse and not model.is_file():
        raise AnalysisError(f"Layer 1 model not found: {model}")
    if not skip_inverse and not binary.is_file():
        raise AnalysisError(f"C filter executable not found: {binary}; run 'make -C Filters'")
    if threshold is not None and (not math.isfinite(threshold) or not 0 < threshold < 1):
        raise AnalysisError("--threshold must be strictly between 0 and 1")
    input_meta = wav_info(source)
    if output.exists():
        raise AnalysisError(f"Output directory exists (refusing to overwrite): {output}")
    output.mkdir(parents=True)
    manifest_path = output / "manifest.json"
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "input": {"path": str(source), "sha256": sha256(source), "wav": input_meta},
        "layer1": {"model_path": str(model), "status": "skipped" if skip_inverse else "pending"},
        "inverse_filter": {"executable": str(binary), "status": "skipped" if skip_inverse else "pending"},
        "layer2": {"model_path": str(l2model), "status": "pending"} if l2model else {"status": "not_requested"},
        "lf": {"model_path": str(lfmodel), "status": "pending", "threshold": lf_threshold}
              if lfmodel else {"status": "not_requested"},
        "mode": "direct_layer2" if skip_inverse else "inverse_then_classification" if (l2model or lfmodel) else "inverse_only",
    }
    def persist() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    persist()
    try:
        if skip_inverse:
            manifest["analysis_audio"] = {"path": str(source), "source": "original_input"}
            persist()
        else:
            # Run from repo root so imports resolve even if caller is elsewhere.
            command = [sys.executable, "-m", "Layer_1", "predict", "--model", str(model), str(source)]
            print("[1/3] Running Layer 1 prediction...", flush=True)
            proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False)
            if proc.returncode != 0:
                raise AnalysisError(f"Layer 1 failed ({proc.returncode}):\n{proc.stderr[-3000:]}\n{proc.stdout[-1000:]}")
            try:
                results = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
            except json.JSONDecodeError as exc:
                raise AnalysisError(f"Layer 1 stdout was not JSON: {proc.stdout[-1000:]}") from exc
            if len(results) != 1 or not isinstance(results[0], dict):
                raise AnalysisError("Expected exactly one JSON prediction from Layer 1")
            prediction = results[0]
            if Path(str(prediction.get("file", ""))).resolve() != source:
                raise AnalysisError("Layer 1 prediction identifies the wrong input file")
            spec, args = validate_spec(prediction.get("spec"))
            (output / "layer1_prediction.json").write_text(json.dumps(prediction, indent=2, allow_nan=False) + "\n")
            manifest["layer1"]["prediction_file"] = "layer1_prediction.json"
            manifest["layer1"]["prediction"] = prediction
            persist()

            reconstructed = output / "reconstructed.wav"
            # Program argument format: type family stage_order hp_hz lp_hz input output menu_option.
            filter_command = [str(binary), *args[:5], str(source), str(reconstructed), args[5]]
            manifest["inverse_filter"].update({
                "normalized_spec": spec, "c_stage_order": int(args[2]),
                "highpass_hz_arg": float(args[3]), "lowpass_hz_arg": float(args[4]),
                "menu_option": int(args[5]), "command": filter_command,
            })
            persist()
            print("[2/3] Running C inverse filter...", flush=True)
            proc = subprocess.run(filter_command, cwd=output, capture_output=True, text=True,
                                  timeout=timeout, check=False)
            manifest["inverse_filter"]["stdout_tail"] = proc.stdout[-3000:]
            manifest["inverse_filter"]["stderr_tail"] = proc.stderr[-3000:]
            if proc.returncode != 0:
                raise AnalysisError(f"C filter failed ({proc.returncode}):\n{proc.stderr[-3000:]}\n{proc.stdout[-1000:]}")
            output_meta = wav_info(reconstructed)
            if output_meta["frames"] != input_meta["frames"]:
                raise AnalysisError(f"C output frame count changed: {input_meta['frames']} -> {output_meta['frames']}")
            manifest["inverse_filter"]["output"] = {
                "path": str(reconstructed), "sha256": sha256(reconstructed), "wav": output_meta
            }
            manifest["inverse_filter"]["status"] = "success"
            manifest["layer1"]["status"] = "success"
            manifest["analysis_audio"] = {"path": str(reconstructed), "source": "reconstructed"}
            persist()
        if lfmodel is not None:
            # A highpass cutoff is needed to choose the lowpass isolation band.
            # The lower edge of an inferred bandpass is its highpass cutoff.
            spec = manifest["inverse_filter"].get("normalized_spec", {})
            kind = spec.get("type")
            if kind not in ("highpass", "bandpass"):
                manifest["lf"].update({"status": "skipped", "reason":
                                       "LF recovery requires a predicted highpass or bandpass filter"})
                persist()
            else:
                estimated_cutoff = float(spec["cutoff"] if kind == "highpass" else spec["cutoff"][0])
                cutoff = lf_cutoff if lf_cutoff is not None else estimated_cutoff
                lfout = output / "lf"
                # Use the existing standalone predictor, so the 27 feature definitions,
                # windowing, saved model format and C lowpass implementation stay identical.
                command = [sys.executable, str(lfscript), "predict", "--input",
                           str(output / "reconstructed.wav"), "--model", str(lfmodel),
                           "--filter-exe", str(binary), "--lp-cutoff", str(cutoff),
                           "--threshold", str(lf_threshold), "--out", str(lfout), "--keep-audio"]
                manifest["lf"].update({"status": "running", "isolation_cutoff_hz": cutoff,
                                       "isolation_cutoff_source": "manual" if lf_cutoff is not None else "layer1",
                                       "predicted_hp_cutoff_hz": estimated_cutoff,
                                       "isolation_order": 4, "command": command})
                persist()
                print(f"[LF] Isolating LF below {cutoff:g} Hz and classifying...", flush=True)
                proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                      timeout=timeout, check=False)
                if proc.returncode != 0:
                    raise AnalysisError(f"LF predictor failed ({proc.returncode}):\n{proc.stderr[-3000:]}\n{proc.stdout[-1000:]}")
                lf_file = lfout / "lf_prediction.json"
                lf_audio = lfout / "lf_isolated.wav"
                if not lf_file.is_file() or not lf_audio.is_file():
                    raise AnalysisError("LF predictor returned success without saving predictions and isolated WAV")
                try:
                    lf_detection = json.loads(lf_file.read_text(encoding="utf-8"))
                except (ValueError, OSError) as exc:
                    raise AnalysisError(f"LF predictor produced invalid JSON: {exc}") from exc
                if (not isinstance(lf_detection, dict)
                    or Path(str(lf_detection.get("input", ""))).resolve() != output / "reconstructed.wav"
                    or not isinstance(lf_detection.get("windows"), list)
                    or not math.isclose(float(lf_detection.get("cutoff_hz", -1)), cutoff, rel_tol=1e-6)):
                    raise AnalysisError("LF predictor returned unexpected input, cutoff, or windows")
                labels = set()
                counts = {}
                for win in lf_detection["windows"]:
                    if not isinstance(win, dict) or not isinstance(win.get("detected"), dict):
                        raise AnalysisError("LF prediction contains invalid windows")
                    labels.update(win["detected"])
                for label in sorted(labels):
                    counts[label] = sum(bool(w["detected"].get(label, False)) for w in lf_detection["windows"])
                ordered_labels = sorted(labels)
                lf_events = lf_events_from_windows(lf_detection["windows"], ordered_labels)
                manifest["lf"].update({"status": "success", "prediction_file": "lf/lf_prediction.json",
                                       "isolated_audio": "lf/lf_isolated.wav", "window_count": len(lf_detection["windows"]),
                                       "positive_windows": counts, "labels": ordered_labels,
                                       "event_count": len(lf_events)})
                persist()

        if l2model is not None:
            target = source if skip_inverse else output / "reconstructed.wav"
            layer2_output = output / "layer2_prediction.json"
            cmd = [sys.executable, "-m", "Layer_2.predict", "--model", str(l2model),
                   "--input", str(target), "--output", str(layer2_output)]
            if threshold is not None:
                cmd.extend(["--threshold", str(threshold)])
            print("[Layer 2] Classifying noise windows...", flush=True)
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                  timeout=timeout, check=False)
            if proc.returncode != 0:
                raise AnalysisError(f"Layer 2 failed ({proc.returncode}):\n{proc.stderr[-3000:]}\n{proc.stdout[-1000:]}")
            if not layer2_output.is_file():
                raise AnalysisError("Layer 2 returned success but did not save layer2_prediction.json")
            try:
                detection = json.loads(layer2_output.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise AnalysisError(f"Layer 2 produced invalid JSON: {exc}") from exc
            if not isinstance(detection, dict) or detection.get("schema_version") != 2:
                raise AnalysisError("Layer 2 produced incompatible prediction JSON schema")
            if Path(str(detection.get("file", ""))).resolve() != target:
                raise AnalysisError("Layer 2 prediction identifies the wrong WAV")
            if not all(k in detection for k in ("events", "windows", "noise_types")):
                raise AnalysisError("Layer 2 result missing windows, events, or noise_types")
            manifest["layer2"].update({"status": "success", "prediction_file": "layer2_prediction.json",
                                      "input_audio": str(target), "window_count": len(detection["windows"]),
                                      "event_count": len(detection["events"]), "threshold": detection.get("threshold")})
            print(f"[Complete] {len(detection['windows'])} windows; {len(detection['events'])} events.", flush=True)
        if l2model is None and lfmodel is None:
            print("[Complete] Saved reconstructed WAV and manifest.", flush=True)
        if l2model is not None or lfmodel is not None:
            target = source if skip_inverse else output / "reconstructed.wav"
            report = {
                "schema_version": 2, "status": "success", "mode": manifest["mode"],
                "original_audio": str(source), "analyzed_audio": str(target),
                "layer1_prediction": manifest["layer1"].get("prediction"),
                "inverse_filter": {"status": manifest["inverse_filter"]["status"],
                                   "parameters": manifest["inverse_filter"].get("normalized_spec")},
                "layer2": ({"noise_types": detection["noise_types"], "events": detection["events"],
                            "window_seconds": detection.get("window_seconds"),
                            "threshold": detection.get("threshold"),
                            "window_results_file": "layer2_prediction.json"}
                           if l2model else {"status": "not_requested"}),
                "lf": ({**{k: v for k, v in manifest["lf"].items() if k != "command"},
                        "events": lf_events} if manifest["lf"]["status"] == "success"
                       else {k: v for k, v in manifest["lf"].items() if k != "command"}),
                "manifest_file": "manifest.json",
            }
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            manifest["report_file"] = "report.json"
        manifest["status"] = "success"
        persist()
        return manifest_path
    except Exception as exc:
        if manifest.get("lf", {}).get("status") == "running":
            manifest["lf"]["status"] = "failed"
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        persist()
        if isinstance(exc, subprocess.TimeoutExpired):
            raise AnalysisError(f"External process timed out after {timeout} seconds") from exc
        raise
