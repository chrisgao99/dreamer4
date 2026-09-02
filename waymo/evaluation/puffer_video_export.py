"""Model-free helpers shared by PufferDrive offline video exporters."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def load_subset_records(
    path: str | Path,
    *,
    sample_start: int,
    num_scenes: int,
) -> list[dict[str, Any]]:
    """Load a contiguous range in the subset's durable sample order."""

    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list):
        raise ValueError(f"Subset manifest has no samples list: {manifest_path}")
    if sample_start < 0:
        raise ValueError(f"sample-start must be non-negative, got {sample_start}")
    if num_scenes < 1:
        raise ValueError(f"num-scenes must be positive, got {num_scenes}")

    ordered = sorted(samples, key=lambda row: int(row["sample_order"]))
    selected = ordered[sample_start : sample_start + num_scenes]
    if len(selected) != num_scenes:
        raise ValueError(
            f"Requested {num_scenes} samples starting at {sample_start}, but "
            f"{manifest_path} only provides {len(selected)}"
        )
    required = {
        "sample_order",
        "dataset_index",
        "scenario_id",
        "focus_track_id",
        "path",
    }
    for row in selected:
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(
                f"Subset sample {row!r} is missing fields: {', '.join(missing)}"
            )
    return [dict(row) for row in selected]


def overlay_label(jpeg: bytes, label: str) -> bytes:
    """Draw a readable identity/progress strip on one Puffer JPEG frame."""

    with Image.open(io.BytesIO(jpeg)) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=18)
    draw.rectangle((0, 0, image.width, 42), fill=(8, 11, 17))
    draw.text((10, 12), str(label), fill=(255, 212, 96), font=font)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=95, optimize=False)
    return output.getvalue()


class FfmpegJpegWriter:
    """Stream concatenated JPEG frames to an atomic H.264 MP4 output."""

    def __init__(
        self,
        output_path: Path,
        *,
        ffmpeg: str,
        fps: float,
        overwrite: bool,
    ) -> None:
        self.output_path = output_path
        self.overwrite = bool(overwrite)
        if output_path.exists() and not self.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}; pass --overwrite to replace it"
            )
        ffmpeg_path = shutil.which(ffmpeg)
        if ffmpeg_path is None:
            raise FileNotFoundError(f"ffmpeg executable was not found: {ffmpeg}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_path = output_path.with_name(
            f".{output_path.stem}.{os.getpid()}.tmp.mp4"
        )
        self.temp_path.unlink(missing_ok=True)
        command = [
            ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "image2pipe",
            "-framerate",
            f"{float(fps):g}",
            "-vcodec",
            "mjpeg",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.temp_path),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self.closed = False

    def write(self, jpeg: bytes) -> None:
        if self.closed or self.process.stdin is None:
            raise RuntimeError("Cannot write to a closed ffmpeg encoder")
        try:
            remaining = memoryview(jpeg)
            while remaining:
                written = self.process.stdin.write(remaining)
                if written is None or written <= 0:
                    raise BrokenPipeError("ffmpeg accepted zero input bytes")
                remaining = remaining[written:]
        except (BrokenPipeError, OSError) as error:
            detail = self._stderr_text()
            raise RuntimeError(f"ffmpeg closed its input early: {detail}") from error

    def _stderr_text(self) -> str:
        if self.process.stderr is None:
            return ""
        return self.process.stderr.read().decode("utf-8", errors="replace").strip()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        close_error: OSError | None = None
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except OSError as error:
                close_error = error
        try:
            return_code = self.process.wait(timeout=120.0)
        except subprocess.TimeoutExpired as error:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)
            self.temp_path.unlink(missing_ok=True)
            raise RuntimeError("ffmpeg did not finish within 120 seconds") from error
        error_text = self._stderr_text()
        if self.process.stderr is not None:
            self.process.stderr.close()
        if close_error is not None or return_code != 0:
            self.temp_path.unlink(missing_ok=True)
            detail = error_text or (
                str(close_error) if close_error is not None else "no diagnostics"
            )
            raise RuntimeError(
                f"ffmpeg exited with code {return_code}: {detail}"
            )
        if self.output_path.exists() and not self.overwrite:
            self.temp_path.unlink(missing_ok=True)
            raise FileExistsError(f"Output appeared while encoding: {self.output_path}")
        self.temp_path.replace(self.output_path)

    def abort(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)
        if self.process.stderr is not None:
            self.process.stderr.close()
        self.temp_path.unlink(missing_ok=True)

    def __enter__(self) -> "FfmpegJpegWriter":
        return self

    def __exit__(self, exc_type: object, *_exc_info: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def preflight_headless_dependencies(
    *,
    ffmpeg: str,
    use_inherited_display: bool,
) -> None:
    """Fail before expensive work when ffmpeg/Xvfb cannot run."""

    if shutil.which(str(ffmpeg)) is None:
        raise FileNotFoundError(f"ffmpeg executable was not found: {ffmpeg}")
    if use_inherited_display:
        return
    if shutil.which("Xvfb") is None:
        raise FileNotFoundError(
            "Xvfb was not found on PATH; PufferDrive headless rendering requires it"
        )
    socket_dir = Path("/tmp/.X11-unix")
    if socket_dir.exists():
        status = socket_dir.stat()
        if status.st_uid != 0:
            mode = status.st_mode & 0o7777
            raise RuntimeError(
                f"{socket_dir} must be owned by root for Xvfb, but uid={status.st_uid} "
                f"mode={mode:o}. Ask the node administrator to restore root:root 1777."
            )
