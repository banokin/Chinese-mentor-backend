"""
Локальный ASR через Hugging Face Transformers (например openai/whisper-large-v3).

Декодирование webm/opus: встроенный ffmpeg из пакета imageio-ffmpeg (не нужен системный ffmpeg в PATH).
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import tempfile
import threading
import wave
from typing import Any

import imageio_ffmpeg
import numpy as np

logger = logging.getLogger(__name__)

_pipe: Any = None
_pipe_lock = threading.Lock()


def _pcm_f32le_mono_16k_args() -> list[str]:
    """Выход: сырой float32 mono 16 kHz в stdout."""
    return [
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "pipe:1",
    ]


def _run_ffmpeg(
    ffmpeg_exe: str,
    input_args: list[str],
    *,
    stdin: bytes | None = None,
) -> tuple[int, bytes, bytes]:
    """Запуск ffmpeg: input_args — всё между общими флагами и выходом PCM."""
    cmd = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *input_args,
        *_pcm_f32le_mono_16k_args(),
    ]
    return subprocess.run(
        cmd,
        input=stdin,
        capture_output=True,
        timeout=120,
        check=False,
    )


def _raw_pcm_to_array(raw: bytes) -> np.ndarray:
    if not raw:
        raise RuntimeError("ffmpeg вернул пустой поток (проверьте запись в браузере)")
    audio = np.frombuffer(raw, dtype=np.float32).copy()
    if audio.size == 0:
        raise RuntimeError("Нулевая длина аудио после декодирования")
    return audio


def _load_audio_16k_mono(data: bytes, filename: str) -> np.ndarray:
    """
    Bytes → float32 mono @ 16 kHz.

    Записи из MediaRecorder (webm/opus) часто плохо читаются с stdin у ffmpeg;
    сначала пишем во временный файл и даём ffmpeg «пробить» контейнер.
    """
    if not data:
        raise RuntimeError("Пустые аудиоданные")

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ext = os.path.splitext(filename)[1].lower() or ".webm"
    if ext not in (".webm", ".wav", ".mp3", ".m4a", ".mp4", ".ogg", ".oga", ".opus"):
        ext = ".webm"

    attempts_err: list[str] = []

    # 1) Временный файл + автоопределение формата (основной путь для Chrome/Safari webm).
    path: str | None = None
    try:
        fd, path = tempfile.mkstemp(suffix=ext)
        os.write(fd, data)
        os.close(fd)

        for input_args in (
            ["-fflags", "+genpts", "-i", path],
            ["-i", path],
            ["-f", "webm", "-i", path],
            ["-f", "matroska", "-i", path],
        ):
            proc = _run_ffmpeg(ffmpeg_exe, input_args)
            if proc.returncode == 0 and proc.stdout:
                return _raw_pcm_to_array(proc.stdout)
            err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
            attempts_err.append(f"{input_args[:4]}… → {err[:300] or proc.returncode}")
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("Декодирование аудио заняло слишком много времени") from e
    except OSError as e:
        raise RuntimeError(f"Не удалось записать временный файл аудио: {e}") from e
    finally:
        if path and os.path.isfile(path):
            try:
                os.unlink(path)
            except OSError:
                pass

    # 2) Fallback: stdin + явный контейнер webm.
    for input_args in (
        ["-fflags", "+genpts", "-f", "webm", "-i", "pipe:0"],
        ["-f", "webm", "-i", "pipe:0"],
        ["-i", "pipe:0"],
    ):
        try:
            proc = _run_ffmpeg(ffmpeg_exe, input_args, stdin=data)
            if proc.returncode == 0 and proc.stdout:
                return _raw_pcm_to_array(proc.stdout)
            err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
            attempts_err.append(f"stdin {input_args[:5]}… → {err[:300] or proc.returncode}")
        except subprocess.TimeoutExpired as e:
            raise RuntimeError("Декодирование аудио заняло слишком много времени") from e

    detail = " | ".join(attempts_err[-4:]) if attempts_err else "нет деталей"
    logger.warning("ffmpeg: все попытки декодирования не удались: %s", detail[:2000])
    raise RuntimeError(
        "Не удалось декодировать аудио (webm/браузер). "
        f"Проверьте, что установлен пакет imageio-ffmpeg. Детали: {detail[:1200]}"
    )


def float32_mono_to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    """float32 mono [-1, 1] → WAV PCM s16le (удобно для HF Inference API)."""
    if audio.size == 0:
        raise RuntimeError("Пустой сигнал для WAV")
    clipped = np.clip(audio.astype(np.float64, copy=False), -1.0, 1.0)
    pcm_i16 = (clipped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_i16.tobytes())
    return buf.getvalue()


def _get_pipeline() -> Any:
    """Ленивая загрузка pipeline (один раз)."""
    global _pipe
    with _pipe_lock:
        if _pipe is not None:
            return _pipe
        import torch
        from transformers import pipeline

        model_id = (os.getenv("HF_WHISPER_MODEL") or "openai/whisper-large-v3").strip()
        token = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or "").strip() or None

        if torch.cuda.is_available():
            dtype = torch.float16
            device: str = "cuda:0"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            # Apple Silicon; при странных ошибках Whisper на MPS переключите на CPU в коде или env.
            dtype = torch.float16
            device = "mps"
        else:
            dtype = torch.float32
            device = "cpu"

        logger.info("Загрузка ASR: %s на %s (%s)", model_id, device, dtype)
        _pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            dtype=dtype,
            device=device,
            token=token,
        )
        return _pipe


def transcribe_waveform_sync(
    audio: np.ndarray,
    *,
    language: str | None = "zh",
) -> tuple[str, dict[str, Any]]:
    """Синхронная транскрипция массива [T] float32 @ 16 kHz."""
    pipe = _get_pipeline()
    model_id = (os.getenv("HF_WHISPER_MODEL") or "").strip()

    lang = (language or "zh").strip()
    # Whisper в HF принимает ISO-код языка (zh) для мандарина.
    gen_kw: dict[str, Any] = {"task": "transcribe", "language": lang}

    # API transformers: waveform + sampling_rate; generate_kwargs → Whisper.
    result = pipe(
        audio,
        sampling_rate=16000,
        generate_kwargs=gen_kw,
    )

    text = ""
    if isinstance(result, dict):
        text = str(result.get("text", "") or "")
    else:
        text = str(result)
    text = text.strip()
    meta: dict[str, Any] = {"model": model_id, "backend": "huggingface", "language": lang}
    return text, meta


def transcribe_bytes_sync(
    data: bytes,
    *,
    filename: str = "audio.webm",
    language: str | None = "zh",
) -> tuple[str, dict[str, Any]]:
    """Полный путь: bytes → ffmpeg (imageio-ffmpeg) → Whisper."""
    audio = _load_audio_16k_mono(data, filename)
    return transcribe_waveform_sync(audio, language=language)
