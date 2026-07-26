"""Analysis API — SSE-based stutter detection."""

import asyncio
import base64
import io
import json
import os
import tempfile

import numpy as np
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import Response, StreamingResponse
from services.detector import get_detector

router = APIRouter(prefix="/api", tags=["analysis"])

LABELS = ["prolongation", "block", "soundrep", "wordrep", "interjection"]

_spectrogram_cache: dict[str, str] = {}


@router.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """Upload audio and stream detection results via SSE."""

    audio_bytes = await file.read()

    ext = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    session_id = f"{os.getpid()}_{id(audio_bytes)}"

    async def event_stream():
        try:
            yield f"event: started\ndata: {json.dumps({'status': 'loading'})}\n\n"
            yield ": ping\n\n"

            detector = get_detector()

            def _load_audio():
                import librosa

                y, _ = librosa.load(tmp_path, sr=detector.sample_rate, mono=True)
                return y.astype(np.float32)

            y = await asyncio.to_thread(_load_audio)

            total_duration = len(y) / detector.sample_rate
            chunk_size = detector.target_length
            total_chunks = int(np.ceil(len(y) / chunk_size))

            chunk_results_all = {}

            for chunk_idx in range(total_chunks):
                start_sample = chunk_idx * chunk_size
                end_sample = min((chunk_idx + 1) * chunk_size, len(y))
                chunk_audio = detector.pad_or_truncate(y[start_sample:end_sample])

                time_start = chunk_idx * detector.target_duration
                time_end = min((chunk_idx + 1) * detector.target_duration, total_duration)

                def _infer_chunk(audio=chunk_audio):
                    results = {}
                    for i in range(5):
                        if detector.models[i] is not None:
                            params = detector.model_params[i]
                            result = detector._predict_model(
                                audio, i, params["n_mels"], params["n_fft"], params["hop_length"]
                            )
                            if result:
                                label_name, prob, is_detected = result
                                results[label_name] = {"probability": prob, "detected": is_detected}
                    return results

                chunk_results = await asyncio.to_thread(_infer_chunk)

                chunk_results_all[chunk_idx] = {
                    "time_start": time_start,
                    "time_end": time_end,
                    "detections": chunk_results,
                }

                aggregated = _aggregate_results(chunk_results_all, chunk_idx + 1)

                event_data = json.dumps(
                    {
                        "chunk": chunk_idx + 1,
                        "total": total_chunks,
                        "time_start": time_start,
                        "time_end": time_end,
                        "results": chunk_results,
                        "aggregated": aggregated,
                    }
                )
                yield f"event: progress\ndata: {event_data}\n\n"
                yield ": ping\n\n"
                await asyncio.sleep(0)

            summary = detector.get_summary(chunk_results_all)

            def _make_spectrogram():
                return _generate_spectrogram(y, detector.sample_rate)

            spectrogram_b64 = await asyncio.to_thread(_make_spectrogram)
            _spectrogram_cache[session_id] = spectrogram_b64

            complete_data = json.dumps(
                {
                    "summary": summary,
                    "total_chunks": total_chunks,
                    "duration": total_duration,
                    "filename": file.filename,
                    "session_id": session_id,
                }
            )
            yield f"event: complete\ndata: {complete_data}\n\n"
            yield ": ping\n\n"

        except Exception as e:
            error_data = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {error_data}\n\n"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/spectrogram/{session_id}")
async def get_spectrogram(session_id: str):
    """Return spectrogram as PNG image."""
    b64 = _spectrogram_cache.pop(session_id, None)
    if not b64:
        return Response(status_code=404)
    img_bytes = base64.b64decode(b64)
    return Response(content=img_bytes, media_type="image/png")


def _aggregate_results(results, processed_chunks):
    """Calculate aggregated stats (matching PyQt5 analysis_widget.py logic)."""
    aggregated = {}
    for label in LABELS:
        max_prob = 0.0
        detected_count = 0

        for chunk_data in results.values():
            if label in chunk_data.get("detections", {}):
                prob = chunk_data["detections"][label]["probability"]
                is_detected = chunk_data["detections"][label]["detected"]
                max_prob = max(max_prob, prob)
                if is_detected and prob > 0.4:
                    detected_count += 1

        confidence = max_prob * 100 if processed_chunks > 0 else 0.0
        detected = max_prob > 0.4

        aggregated[label] = {
            "confidence": confidence,
            "detected": detected,
            "max_probability": max_prob,
            "detected_chunks": detected_count,
        }

    return aggregated


def _generate_spectrogram(y, sample_rate):
    """Generate spectrogram image as base64 PNG (matching PyQt5 plot_canvas.py)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 2.5), facecolor="#1e1e1e")
    ax.set_facecolor("#1e1e1e")

    ax.specgram(y, Fs=sample_rate, cmap="viridis", NFFT=1024, noverlap=512)
    ax.set_xlabel("Time (s)", color="#ffffff")
    ax.set_ylabel("Frequency (Hz)", color="#ffffff")
    ax.set_title("Spectrogram", color="#ffffff")
    ax.tick_params(axis="x", colors="#ffffff")
    ax.tick_params(axis="y", colors="#ffffff")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=72, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")
