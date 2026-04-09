import os
import shutil
import subprocess
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# Keep these in sync with extract_circles.py
root = '/Volumes/Shared Bartololab3/Remi/TrackingOnTheFly/'
path = '/'
name = 'started_buckled'
ext = '.avi'


def build_movie_path() -> str:
    directory = os.path.join(root, path.lstrip('/'))
    return os.path.join(directory, f"{name}{ext}")


def frame_stats(frame: np.ndarray) -> str:
    nonzero = int(np.count_nonzero(frame))
    total = int(frame.size)
    nonzero_pct = 100.0 * nonzero / max(total, 1)
    return (
        f"shape={frame.shape}, dtype={frame.dtype}, "
        f"min={frame.min()}, max={frame.max()}, mean={frame.mean():.2f}, "
        f"std={frame.std():.2f}, nonzero={nonzero_pct:.3f}%"
    )


def normalize_linear(gray: np.ndarray) -> np.ndarray:
    gmin = float(gray.min())
    gmax = float(gray.max())
    if gmax <= gmin:
        return np.zeros_like(gray, dtype=np.uint8)
    return ((gray.astype(np.float32) - gmin) * (255.0 / (gmax - gmin))).astype(np.uint8)


def normalize_percentile(gray: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> np.ndarray:
    p_lo, p_hi = np.percentile(gray, [lo, hi])
    if p_hi <= p_lo:
        return normalize_linear(gray)
    out = np.clip((gray.astype(np.float32) - p_lo) * (255.0 / (p_hi - p_lo)), 0, 255)
    return out.astype(np.uint8)


def normalize_clahe(gray: np.ndarray) -> np.ndarray:
    base = normalize_percentile(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(base)


def as_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    return frame.squeeze()


def to_display(frame: np.ndarray, mode: str) -> np.ndarray:
    gray = as_gray(frame)
    if gray.dtype != np.uint8:
        gray = normalize_linear(gray)

    if mode == "raw8":
        return gray
    if mode == "linear":
        return normalize_linear(gray)
    if mode == "percentile":
        return normalize_percentile(gray)
    if mode == "clahe":
        return normalize_clahe(gray)
    return gray


def ffprobe_stream(movie_path: str) -> Dict[str, str]:
    if shutil.which("ffprobe") is None:
        return {}

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-select_streams",
        "v:0",
        movie_path,
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        return {}

    out: Dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


class PacketStreamReader:
    """Read one video packet per frame as gray indices using ffmpeg stream copy."""

    def __init__(self, movie_path: str, width: int, height: int):
        self.width = int(width)
        self.height = int(height)
        self.frame_size = self.width * self.height
        self.proc: Optional[subprocess.Popen] = None

        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            movie_path,
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-f",
            "rawvideo",
            "-",
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8,
        )

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.proc is None or self.proc.stdout is None:
            return False, None
        payload = self.proc.stdout.read(self.frame_size)
        if payload is None or len(payload) < self.frame_size:
            return False, None
        frame = np.frombuffer(payload, dtype=np.uint8).reshape((self.height, self.width))
        return True, frame

    def release(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdout is not None:
                self.proc.stdout.close()
            if self.proc.stderr is not None:
                self.proc.stderr.close()
            self.proc.terminate()
        except Exception:
            pass
        self.proc = None


def main() -> None:
    movie_path = build_movie_path()
    print(f"Movie path: {movie_path}")
    if not os.path.exists(movie_path):
        print("ERROR: file does not exist.")
        return

    print(f"OpenCV version: {cv2.__version__}")
    stream = ffprobe_stream(movie_path)
    if stream:
        print("\nffprobe:")
        for k in ["codec_name", "codec_tag_string", "width", "height", "pix_fmt", "r_frame_rate", "bits_per_raw_sample"]:
            if k in stream:
                print(f"  {k}={stream[k]}")

    cap = cv2.VideoCapture(movie_path)
    if not cap.isOpened():
        print("ERROR: OpenCV could not open movie")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_txt = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    print("\nOpenCV stream:")
    print(f"  frame_count={count}, size={width}x{height}, fps={fps}, fourcc={fourcc_txt!r}")

    packet_reader = None
    if shutil.which("ffmpeg") is not None:
        try:
            packet_reader = PacketStreamReader(movie_path, width, height)
            ok_p, fr_p = packet_reader.read()
            if ok_p and fr_p is not None:
                print(f"\nPacket stream first frame: {frame_stats(fr_p)}")
                os.makedirs("debug_frames", exist_ok=True)
                cv2.imwrite("debug_frames/packet_first_raw8.png", fr_p)
                cv2.imwrite("debug_frames/packet_first_norm.png", normalize_percentile(fr_p))
                packet_reader.release()
                packet_reader = PacketStreamReader(movie_path, width, height)
            else:
                print("\nPacket stream first frame read FAILED")
                if packet_reader is not None:
                    packet_reader.release()
                    packet_reader = None
        except Exception as exc:
            print(f"\nPacket stream reader init failed: {exc}")
            packet_reader = None

    print("\nControls:")
    print("  q: quit")
    print("  space: pause")
    print("  s: switch source (opencv <-> packet)")
    print("  n: cycle normalization (raw8 -> linear -> percentile -> clahe)")
    print("  r: force raw8 mode")

    source = "packet" if packet_reader is not None else "opencv"
    norm_modes = ["raw8", "linear", "percentile", "clahe"]
    norm_idx = 0
    paused = False
    idx = -1

    while True:
        if not paused:
            if source == "opencv":
                ok, frame = cap.read()
                gray = as_gray(frame) if ok and frame is not None else None
            else:
                ok, frame = packet_reader.read() if packet_reader is not None else (False, None)
                gray = frame if ok else None

            if not ok or gray is None:
                print(f"End/fail on source={source}")
                break

            idx += 1
            vis = to_display(gray, norm_modes[norm_idx])
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            label = f"src={source} frame={idx} norm={norm_modes[norm_idx]}"
            cv2.putText(vis_bgr, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

            if idx < 10 or idx % 100 == 0:
                print(f"{label}: {frame_stats(gray)}")
                p = np.percentile(gray, [1, 50, 99])
                print(f"  percentiles p01/p50/p99: {p[0]:.2f}/{p[1]:.2f}/{p[2]:.2f}")

            cv2.imshow("movie_debug", vis_bgr)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
        if key == ord("n"):
            norm_idx = (norm_idx + 1) % len(norm_modes)
            print(f"Normalization mode: {norm_modes[norm_idx]}")
        if key == ord("r"):
            norm_idx = 0
            print("Normalization mode: raw8")
        if key == ord("s"):
            if source == "opencv" and packet_reader is not None:
                source = "packet"
            else:
                source = "opencv"
            print(f"Switched source to: {source}")

    cap.release()
    if packet_reader is not None:
        packet_reader.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
