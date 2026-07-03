"""
Learning-free drone detection v2 — (x,y,t) voxel + 3D CC.

Pipeline (per 33ms window):
  Stage 0: EMA baseline subtraction
  Stage 1: Residual event extraction
  Stage 2: 100ms voxel grid (5px × 1ms)
  Stage 3: 3D connected component (6-connectivity)
  Stage 4: Component feature filtering (duration / displacement / size)

Usage:
  python3 detect_v2.py --seqs 81 82
  python3 detect_v2.py --seq 82 --vis --max-windows 200
"""

import os
import glob
import argparse
import csv
import datetime
import subprocess
import numpy as np
import h5py
import hdf5plugin
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import label as scipy_label, find_objects
from tqdm import tqdm

# 비디오 출력 해상도 (좌 패널만)
VIDEO_W = 1280
VIDEO_H = 720

# ── Sensor ────────────────────────────────────────────────────────────────────
SENSOR_W  = 1280
SENSOR_H  = 720
WINDOW_US = 33_333

# ── V2 파라미터 (검증 완료) ────────────────────────────────────────────────────
GRID_PX    = 5
T_CELL_US  = 1_000
BUFFER_WIN = 3
T_CELLS    = 100
ALPHA      = 0.2
DUR_MIN    = 30    # ms
DISP_MIN   = 5     # px
SIZE_MIN   = 5     # px
SIZE_MAX   = 200   # px
WARM_UP    = 100

GRID_W = SENSOR_W // GRID_PX + 1  # 257
GRID_H = SENSOR_H // GRID_PX + 1  # 145

STRUCT_6 = np.zeros((3, 3, 3), dtype=int)
STRUCT_6[1, 1, 1] = 1
STRUCT_6[0, 1, 1] = 1; STRUCT_6[2, 1, 1] = 1
STRUCT_6[1, 0, 1] = 1; STRUCT_6[1, 2, 1] = 1
STRUCT_6[1, 1, 0] = 1; STRUCT_6[1, 1, 2] = 1

# ── Config ────────────────────────────────────────────────────────────────────
_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "detect_v2.yaml")

def load_config(config_path):
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    d = cfg.get("DATA", {})
    p = cfg.get("PIPELINE", {})
    v = cfg.get("VIS", {})
    return argparse.Namespace(
        root             = d.get("root", "/home/c132/CV2026/FRED"),
        split            = d.get("split", "train"),
        seqs             = d.get("seqs", [81, 82]),
        track_max_dist   = 100,
        track_min_frames = 1,
        out_dir          = v.get("out_dir", "log/detect_v2"),
        vis_interval     = v.get("interval", 30),
        buffer_win       = p.get("buffer_win", 3),
        alpha            = p.get("alpha", 0.2),
        dur_min          = p.get("dur_min", 30),
        disp_min         = p.get("disp_min", 5),
        adaptive_disp    = p.get("adaptive_disp", False),
        size_min         = p.get("size_min", 5),
        size_max         = p.get("size_max", 200),
        age_min          = p.get("age_min", 1),
        max_miss         = p.get("max_miss", 0),
    )


# ── Data loading (detect_learning_free.py와 동일) ─────────────────────────────

def load_coords(coord_path):
    coords = {}
    if not os.path.exists(coord_path):
        return coords
    with open(coord_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ts_str, rest = line.split(": ", 1)
                parts = rest.split(", ")
                x1, y1, x2, y2 = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                drone_id = int(parts[4]) if len(parts) > 4 else 0
                ts_us = int(float(ts_str) * 1e6)
                coords.setdefault(ts_us, []).append((x1, y1, x2, y2, drone_id))
            except Exception:
                continue
    return coords


def get_gt_bboxes(coords, t_start, t_end):
    if not coords:
        return []
    t_center = (t_start + t_end) // 2
    best_ts = min(coords.keys(), key=lambda ts: abs(ts - t_center))
    if abs(best_ts - t_center) < WINDOW_US:
        return [(x1, y1, x2, y2) for x1, y1, x2, y2, *_ in coords[best_ts]]
    return []


def get_gt_bboxes_with_id(coords, t_start, t_end):
    if not coords:
        return []
    t_center = (t_start + t_end) // 2
    best_ts = min(coords.keys(), key=lambda ts: abs(ts - t_center))
    if abs(best_ts - t_center) < WINDOW_US:
        return list(coords[best_ts])  # (x1, y1, x2, y2, drone_id)
    return []


def count_fa_blobs(fp_x, fp_y):
    if len(fp_x) == 0:
        return 0
    frame = np.zeros((SENSOR_H, SENSOR_W), dtype=np.uint8)
    xi = fp_x.astype(int).clip(0, SENSOR_W - 1)
    yi = fp_y.astype(int).clip(0, SENSOR_H - 1)
    frame[yi, xi] = 1
    struct2d = np.ones((3, 3), dtype=int)
    _, n = scipy_label(frame, structure=struct2d)
    return n


def build_drone_tracks(coords, t_min):
    tracks = {}
    for ts_us, bboxes in coords.items():
        win_idx = (ts_us - t_min) // WINDOW_US
        for x1, y1, x2, y2, did in bboxes:
            if did not in tracks:
                tracks[did] = {}
            if win_idx not in tracks[did]:
                tracks[did][win_idx] = (x1, y1, x2, y2)
    return tracks


# ── Evaluation (detect_learning_free.py와 동일) ───────────────────────────────

def compute_iou(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def evaluate_window(pred_bboxes, gt_bboxes, iou_thresh=0.5):
    if not gt_bboxes:
        return 0, len(pred_bboxes), 0
    if not pred_bboxes:
        return 0, 0, len(gt_bboxes)
    gt_matched = [False] * len(gt_bboxes)
    tp = fp = 0
    for pred in pred_bboxes:
        matched = False
        for gi, gt in enumerate(gt_bboxes):
            if not gt_matched[gi] and compute_iou(pred, gt) >= iou_thresh:
                gt_matched[gi] = True
                matched = True
                break
        if matched:
            tp += 1
        else:
            fp += 1
    fn = sum(1 for m in gt_matched if not m)
    return tp, fp, fn


def evaluate_tracklets(completed_tracklets, drone_tracks, min_len,
                        iou_thresh=0.5, coverage_thresh=0.3):
    long_tracklets = [t for t in completed_tracklets if len(t) >= min_len]
    if not long_tracklets or not drone_tracks:
        return 0.0, 0.0, 0, len(long_tracklets)
    drone_detected = {did: False for did in drone_tracks}
    tp = fp = 0
    for tracklet in long_tracklets:
        best_frac, best_did = 0.0, None
        for did, drone_frames in drone_tracks.items():
            overlap = sum(
                1 for win_idx, x1, y1, x2, y2 in tracklet
                if win_idx in drone_frames
                and compute_iou((x1, y1, x2, y2), drone_frames[win_idx]) >= iou_thresh
            )
            frac = overlap / len(tracklet)
            if frac > best_frac:
                best_frac, best_did = frac, did
        if best_frac >= coverage_thresh:
            tp += 1
            drone_detected[best_did] = True
        else:
            fp += 1
    n_drones = len(drone_detected)
    return (
        tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        sum(drone_detected.values()) / n_drones if n_drones > 0 else 0.0,
        tp, fp,
    )


# ── V2 stage 2~4 ──────────────────────────────────────────────────────────────

def build_voxel_and_detect(res_buffer, t_start, buffer_win, dur_min, disp_min,
                           adaptive_disp, size_min, size_max):
    """
    Stage 2: buffer_win × 33ms residual events → voxel grid
    Stage 3: 3D CC (6-connectivity)
    Stage 4: feature 추출 + 조건 판정 → confirmed bbox list
    Returns list of (x1, y1, x2, y2) in sensor pixel coords.
    """
    t_cells = buffer_win * WINDOW_US // T_CELL_US
    t_buf_start = t_start - (buffer_win - 1) * WINDOW_US

    ax_ = np.concatenate([b[0] for b in res_buffer])
    ay_ = np.concatenate([b[1] for b in res_buffer])
    at_ = np.concatenate([b[2] for b in res_buffer])

    if len(ax_) == 0:
        return []

    vx = (ax_ / GRID_PX).astype(int).clip(0, GRID_W - 1)
    vy = (ay_ / GRID_PX).astype(int).clip(0, GRID_H - 1)
    vt = ((at_ - t_buf_start) // T_CELL_US).astype(int).clip(0, t_cells - 1)

    voxel_grid = np.zeros((GRID_H, GRID_W, t_cells), dtype=np.bool_)
    voxel_grid[vy, vx, vt] = True

    labeled, n_comp = scipy_label(voxel_grid, structure=STRUCT_6)
    if n_comp == 0:
        return []

    confirmed = []
    objects = find_objects(labeled)
    for cid, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        sub = labeled[sl]
        local_vox = np.argwhere(sub == cid)
        if len(local_vox) < 3:
            continue

        y_off = sl[0].start; x_off = sl[1].start; t_off = sl[2].start
        ys = local_vox[:, 0] + y_off
        xs = local_vox[:, 1] + x_off
        ts = local_vox[:, 2] + t_off

        t_min_c = int(ts.min()); t_max_c = int(ts.max())
        duration = t_max_c - t_min_c  # ms

        early = local_vox[ts == t_min_c]; late = local_vox[ts == t_max_c]
        cx_e = float(early[:, 1].mean()); cy_e = float(early[:, 0].mean())
        cx_l = float(late[:, 1].mean());  cy_l = float(late[:, 0].mean())
        displacement = ((cx_l - cx_e) ** 2 + (cy_l - cy_e) ** 2) ** 0.5 * GRID_PX

        width  = int((xs.max() - xs.min() + 1) * GRID_PX)
        height = int((ys.max() - ys.min() + 1) * GRID_PX)

        eff_disp_min = max(1, min(width, height) * 0.05) if adaptive_disp else disp_min

        if (duration >= dur_min and displacement >= eff_disp_min
                and size_min <= width <= size_max and size_min <= height <= size_max):
            bx1 = int(xs.min()) * GRID_PX
            by1 = int(ys.min()) * GRID_PX
            bx2 = int(xs.max() + 1) * GRID_PX
            by2 = int(ys.max() + 1) * GRID_PX
            confirmed.append((bx1, by1, bx2, by2))

    return confirmed


# ── Simplified tracklet matching ──────────────────────────────────────────────

def match_blobs(new_blobs, blob_history, track_max_dist, next_tid, max_miss=0):
    """
    Greedy exclusive 1:1 matching by centroid distance.
    blob_history: [(x1,y1,x2,y2, age, tid, miss), ...]
    Returns: new_history, dead_tids, next_tid
    """
    if not new_blobs and not blob_history:
        return [], [], next_tid

    if not new_blobs:
        new_history = []
        dead_tids = []
        for h in blob_history:
            miss = h[6] + 1
            if miss > max_miss:
                dead_tids.append(h[5])
            else:
                new_history.append((h[0], h[1], h[2], h[3], h[4], h[5], miss))
        return new_history, dead_tids, next_tid

    if not blob_history:
        new_history = []
        for (x1, y1, x2, y2) in new_blobs:
            new_history.append((x1, y1, x2, y2, 1, next_tid, 0))
            next_tid += 1
        return new_history, [], next_tid

    new_cx = [(x1 + x2) / 2 for x1, y1, x2, y2 in new_blobs]
    new_cy = [(y1 + y2) / 2 for x1, y1, x2, y2 in new_blobs]
    hist_cx = [(h[0] + h[2]) / 2 for h in blob_history]
    hist_cy = [(h[1] + h[3]) / 2 for h in blob_history]

    dists = []
    for ni in range(len(new_blobs)):
        for hi in range(len(blob_history)):
            d = ((new_cx[ni] - hist_cx[hi]) ** 2 + (new_cy[ni] - hist_cy[hi]) ** 2) ** 0.5
            if d <= track_max_dist:
                dists.append((d, ni, hi))
    dists.sort()

    used_ni = set(); used_hi = set()
    matched = {}
    for d, ni, hi in dists:
        if ni not in used_ni and hi not in used_hi:
            matched[ni] = hi
            used_ni.add(ni); used_hi.add(hi)

    new_history = []
    for ni, (x1, y1, x2, y2) in enumerate(new_blobs):
        if ni in matched:
            h = blob_history[matched[ni]]
            new_history.append((x1, y1, x2, y2, h[4] + 1, h[5], 0))
        else:
            new_history.append((x1, y1, x2, y2, 1, next_tid, 0))
            next_tid += 1

    dead_tids = []
    for hi, h in enumerate(blob_history):
        if hi not in used_hi:
            miss = h[6] + 1
            if miss > max_miss:
                dead_tids.append(h[5])
            else:
                new_history.append((h[0], h[1], h[2], h[3], h[4], h[5], miss))

    return new_history, dead_tids, next_tid


# ── Visualization ─────────────────────────────────────────────────────────────

def find_rgb(rgb_dir, win_idx, total_windows):
    jpgs = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))
    if not jpgs:
        return None
    frac = win_idx / max(total_windows - 1, 1)
    idx = int(frac * (len(jpgs) - 1))
    return jpgs[min(idx, len(jpgs) - 1)]


def visualize_v2(x, y, pred_bboxes, gt_bboxes,
                 rgb_path, seq_id, win_idx, tp, fp, fn, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(22, 8))
    fig.patch.set_facecolor("#111111")
    for ax in axes:
        ax.set_facecolor("black")

    # Left: raw events + predictions
    ax = axes[0]
    n = len(x)
    cap = min(n, 60_000)
    idx = np.random.choice(n, cap, replace=False) if n > cap else np.arange(n)
    ax.scatter(x[idx], y[idx], s=0.2, c="#555555", alpha=0.5, rasterized=True)

    for b in gt_bboxes:
        ax.add_patch(patches.Rectangle(
            (b[0], b[1]), b[2] - b[0], b[3] - b[1],
            fill=False, edgecolor="lime", lw=2))
    for b in pred_bboxes:
        ax.add_patch(patches.Rectangle(
            (b[0], b[1]), b[2] - b[0], b[3] - b[1],
            fill=False, edgecolor="cyan", lw=2))

    ax.set_xlim(0, SENSOR_W); ax.set_ylim(SENSOR_H, 0)
    ax.set_aspect("equal"); ax.tick_params(colors="white")
    ax.set_title(
        f"Events  pred={len(pred_bboxes)}  GT=lime  Pred=cyan\n"
        f"TP={tp}  FP={fp}  FN={fn}",
        color="white", fontsize=9)

    # Right: RGB
    ax = axes[1]
    if rgb_path and os.path.exists(rgb_path):
        from PIL import Image
        img = np.array(Image.open(rgb_path))
        ax.imshow(img)
        ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0)
    else:
        ax.set_xlim(0, SENSOR_W); ax.set_ylim(SENSOR_H, 0)
        ax.text(SENSOR_W / 2, SENSOR_H / 2, "No RGB",
                color="white", ha="center", va="center", fontsize=14)
    for b in gt_bboxes:
        ax.add_patch(patches.Rectangle(
            (b[0], b[1]), b[2] - b[0], b[3] - b[1],
            fill=False, edgecolor="lime", lw=2))

    ax.set_aspect("equal"); ax.tick_params(colors="white")
    ax.set_title(f"RGB  GT=lime  seq={seq_id}  win={win_idx}", color="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


# ── Video frame renderer (좌 패널만) ─────────────────────────────────────────

def render_left_frame(x, y, pred_bboxes, gt_bboxes, win_idx):
    """
    좌 패널(raw events + GT + pred bbox)을 VIDEO_W×VIDEO_H RGB 배열로 반환.
    """
    dpi = 100
    fig, ax = plt.subplots(1, 1,
                            figsize=(VIDEO_W / dpi, VIDEO_H / dpi),
                            dpi=dpi)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    n = len(x)
    cap = min(n, 60_000)
    idx = np.random.choice(n, cap, replace=False) if n > cap else np.arange(n)
    ax.scatter(x[idx], y[idx], s=0.2, c="#555555", alpha=0.5, rasterized=True)

    for b in gt_bboxes:
        ax.add_patch(patches.Rectangle(
            (b[0], b[1]), b[2] - b[0], b[3] - b[1],
            fill=False, edgecolor="lime", lw=2))
    for b in pred_bboxes:
        ax.add_patch(patches.Rectangle(
            (b[0], b[1]), b[2] - b[0], b[3] - b[1],
            fill=False, edgecolor="cyan", lw=2))

    ax.set_xlim(0, SENSOR_W); ax.set_ylim(SENSOR_H, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]  # RGB
    plt.close(fig)
    return buf


def run_sequence_to_video(seq_id, cfg, out_path, max_windows=None, with_rgb=False):
    """매 윈도우 좌 패널 프레임을 ffmpeg에 파이프 → mp4 저장. 평가 없음."""
    seq_dir    = os.path.join(cfg.root, cfg.split, str(seq_id))
    hdf5_path  = os.path.join(seq_dir, "Event", "events.hdf5")
    coord_path = os.path.join(seq_dir, "coordinates.txt")

    if not os.path.exists(hdf5_path):
        print(f"[SKIP] seq {seq_id}: HDF5 not found")
        return

    coords = load_coords(coord_path)

    with h5py.File(hdf5_path, "r") as f:
        indexes = f["CD"]["indexes"][:]
        offset  = int(f["CD"]["indexes"].attrs.get("offset", 0))

    ts_arr = indexes["ts"].astype(np.int64) + offset
    id_arr = indexes["id"].copy()
    t_min  = int(ts_arr[ts_arr >= 0].min())
    t_max  = int(ts_arr.max())

    total_windows  = (t_max - t_min) // WINDOW_US
    window_indices = list(range(min(total_windows, max_windows) if max_windows else total_windows))

    baseline   = np.zeros((GRID_H, GRID_W), dtype=np.float64)
    res_buffer = []
    rgb_dir    = os.path.join(seq_dir, "RGB") if with_rgb else None

    out_w = VIDEO_W * 2 if with_rgb else VIDEO_W
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{out_w}x{VIDEO_H}",
        "-framerate", "30",
        "-i", "pipe:0",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        out_path,
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    print(f"seq {seq_id}: {len(window_indices)} windows → {out_path}")

    with h5py.File(hdf5_path, "r") as fh:
        for win_idx in tqdm(window_indices, desc=f"seq {seq_id} video", unit="win"):
            t_start = t_min + win_idx * WINDOW_US
            t_end   = t_start + WINDOW_US

            i_s  = int(np.searchsorted(ts_arr, t_start))
            i_e  = int(np.searchsorted(ts_arr, t_end))
            ev_s = int(id_arr[i_s]) if i_s < len(id_arr) else 0
            ev_e = int(id_arr[i_e]) if i_e < len(id_arr) else int(id_arr[-1])

            gt_bboxes = get_gt_bboxes(coords, t_start, t_end)

            if ev_e <= ev_s:
                res_buffer.append((np.array([]), np.array([]), np.array([])))
                if len(res_buffer) > cfg.buffer_win:
                    res_buffer.pop(0)
                frame = render_left_frame(
                    np.array([]), np.array([]), [], gt_bboxes, win_idx)
                proc.stdin.write(frame.tobytes())
                continue

            data = fh["CD"]["events"][ev_s:ev_e]
            x = data["x"].astype(np.float32)
            y = data["y"].astype(np.float32)
            t = data["t"].astype(np.int64) + offset
            del data

            # Stage 0: EMA baseline
            gx = (x / GRID_PX).astype(int).clip(0, GRID_W - 1)
            gy = (y / GRID_PX).astype(int).clip(0, GRID_H - 1)
            cell_count = np.zeros((GRID_H, GRID_W), dtype=np.float32)
            np.add.at(cell_count, (gy, gx), 1)
            if win_idx == 0:
                baseline[:] = cell_count
            else:
                baseline = cfg.alpha * cell_count + (1 - cfg.alpha) * baseline

            # Stage 1: residual
            active = (cell_count - baseline) > 0
            mask   = active[gy, gx]
            res_buffer.append((x[mask], y[mask], t[mask]))
            if len(res_buffer) > cfg.buffer_win:
                res_buffer.pop(0)

            if win_idx < WARM_UP or len(res_buffer) < cfg.buffer_win:
                confirmed = []
            else:
                confirmed = build_voxel_and_detect(
                    res_buffer, t_start,
                    cfg.buffer_win, cfg.dur_min, cfg.disp_min,
                    cfg.adaptive_disp, cfg.size_min, cfg.size_max)

            ev_frame = render_left_frame(x, y, confirmed, gt_bboxes, win_idx)
            if with_rgb and rgb_dir:
                from PIL import Image as PILImage
                rgb_path = find_rgb(rgb_dir, win_idx, len(window_indices))
                if rgb_path and os.path.exists(rgb_path):
                    rgb_img = np.array(PILImage.open(rgb_path).resize((VIDEO_W, VIDEO_H)))
                else:
                    rgb_img = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
                frame = np.concatenate([rgb_img, ev_frame], axis=1)
            else:
                frame = ev_frame
            proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()
    print(f"저장 완료: {out_path}")


# ── Sequence runner ───────────────────────────────────────────────────────────

def run_sequence(seq_id, cfg, vis=False, vis_only=False, max_windows=None, null_test=False, save_video=False, vid_prefix=""):
    seq_dir   = os.path.join(cfg.root, cfg.split, str(seq_id))
    hdf5_path = os.path.join(seq_dir, "Event", "events.hdf5")
    coord_path = os.path.join(seq_dir, "coordinates.txt")
    rgb_dir   = os.path.join(seq_dir, "RGB")

    if not os.path.exists(hdf5_path):
        print(f"[SKIP] seq {seq_id}: HDF5 not found")
        return None

    coords = load_coords(coord_path)

    with h5py.File(hdf5_path, "r") as f:
        indexes = f["CD"]["indexes"][:]
        offset  = int(f["CD"]["indexes"].attrs.get("offset", 0))

    ts_arr = indexes["ts"].astype(np.int64) + offset
    id_arr = indexes["id"].copy()
    t_min  = int(ts_arr[ts_arr >= 0].min())
    t_max  = int(ts_arr.max())

    total_windows  = (t_max - t_min) // WINDOW_US
    window_indices = list(range(min(total_windows, max_windows) if max_windows else total_windows))

    vis_dir = None
    if vis or vis_only:
        vis_dir = os.path.join(cfg.out_dir, f"seq_{seq_id}_vis")
        os.makedirs(vis_dir, exist_ok=True)

    drone_tracks = build_drone_tracks(coords, t_min)

    baseline         = np.zeros((GRID_H, GRID_W), dtype=np.float64)
    res_buffer       = []
    blob_history     = []
    next_tid         = 0
    active_tracklets = {}
    completed_tracklets = []

    # video 저장
    vid_proc = None
    if save_video:
        vid_path = os.path.join(cfg.out_dir, f"{vid_prefix}_seq{seq_id}.mp4")
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{VIDEO_W}x{VIDEO_H}",
            "-framerate", "30",
            "-i", "pipe:0",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            vid_path,
        ]
        vid_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    # event-level Pd/Fa 누적
    EV_CORRECT_THRESH = 0.5
    obj_stats   = {}   # (drone_id, win_idx) -> [target_count, predicted_count]
    fa_blobs    = 0
    ev_frame_count = 0

    total_tp = total_fp = total_fn = 0
    gt_windows = covered_windows = 0

    print(f"seq {seq_id}: {len(window_indices)} windows")

    with h5py.File(hdf5_path, "r") as fh:
        pbar = tqdm(window_indices, desc=f"seq {seq_id}", unit="win",
                    dynamic_ncols=True, leave=True)
        for win_idx in pbar:
            t_start = t_min + win_idx * WINDOW_US
            t_end   = t_start + WINDOW_US

            i_s  = int(np.searchsorted(ts_arr, t_start))
            i_e  = int(np.searchsorted(ts_arr, t_end))
            ev_s = int(id_arr[i_s]) if i_s < len(id_arr) else 0
            ev_e = int(id_arr[i_e]) if i_e < len(id_arr) else int(id_arr[-1])

            gt_bboxes = get_gt_bboxes(coords, t_start, t_end)

            if ev_e <= ev_s:
                res_buffer.append((np.array([]), np.array([]), np.array([])))
                if len(res_buffer) > cfg.buffer_win:
                    res_buffer.pop(0)
                tp, fp, fn = evaluate_window([], gt_bboxes)
                total_tp += tp; total_fp += fp; total_fn += fn
                if gt_bboxes:
                    gt_windows += 1
                continue

            data = fh["CD"]["events"][ev_s:ev_e]
            x = data["x"].astype(np.float32)
            y = data["y"].astype(np.float32)
            t = data["t"].astype(np.int64) + offset
            del data

            # Stage 0: EMA baseline
            gx = (x / GRID_PX).astype(int).clip(0, GRID_W - 1)
            gy = (y / GRID_PX).astype(int).clip(0, GRID_H - 1)
            cell_count = np.zeros((GRID_H, GRID_W), dtype=np.float32)
            np.add.at(cell_count, (gy, gx), 1)

            if win_idx == 0:
                baseline[:] = cell_count
            else:
                baseline = cfg.alpha * cell_count + (1 - cfg.alpha) * baseline

            # Stage 1: residual events
            active = (cell_count - baseline) > 0
            mask   = active[gy, gx]
            res_buffer.append((x[mask], y[mask], t[mask]))
            if len(res_buffer) > cfg.buffer_win:
                res_buffer.pop(0)

            # Warm-up 또는 버퍼 미충족
            if win_idx < WARM_UP or len(res_buffer) < cfg.buffer_win:
                tp, fp, fn = evaluate_window([], gt_bboxes)
                total_tp += tp; total_fp += fp; total_fn += fn
                if gt_bboxes:
                    gt_windows += 1
                continue

            # Stage 2~4
            candidates = build_voxel_and_detect(
                res_buffer, t_start,
                cfg.buffer_win, cfg.dur_min, cfg.disp_min,
                cfg.adaptive_disp, cfg.size_min, cfg.size_max)

            # Tracklet matching (age 누적)
            blob_history, dead_tids, next_tid = match_blobs(
                candidates, blob_history, cfg.track_max_dist, next_tid, cfg.max_miss)

            for h in blob_history:
                tid = h[5]
                if tid not in active_tracklets:
                    active_tracklets[tid] = []
                active_tracklets[tid].append((win_idx, h[0], h[1], h[2], h[3]))
            for tid in dead_tids:
                if tid in active_tracklets:
                    completed_tracklets.append(active_tracklets.pop(tid))

            # age >= age_min인 blob만 confirmed로 출력
            confirmed = [(h[0], h[1], h[2], h[3]) for h in blob_history if h[4] >= cfg.age_min]

            if vis_only:
                tp = fp = fn = 0
            else:
                # Window-level evaluation
                tp, fp, fn = evaluate_window(confirmed, gt_bboxes)
                total_tp += tp; total_fp += fp; total_fn += fn
                if gt_bboxes:
                    gt_windows += 1
                    if tp > 0:
                        covered_windows += 1

                # Event-level Pd/Fa
                gt_with_id = get_gt_bboxes_with_id(coords, t_start, t_end)
                if null_test and gt_with_id:
                    shuffled = []
                    for bx1, by1, bx2, by2, did in gt_with_id:
                        w = bx2 - bx1; h = by2 - by1
                        cx = np.random.randint(w // 2, SENSOR_W - w // 2)
                        cy = np.random.randint(h // 2, SENSOR_H - h // 2)
                        shuffled.append((cx - w//2, cy - h//2, cx + w//2, cy + h//2, did))
                    gt_with_id = shuffled
                if len(x) > 0:
                    # target labeling
                    target_id = np.zeros(len(x), dtype=np.int32)
                    for bx1, by1, bx2, by2, did in gt_with_id:
                        m = (x >= bx1) & (x <= bx2) & (y >= by1) & (y <= by2)
                        target_id[m] = did

                    # predicted labeling
                    is_pred = np.zeros(len(x), dtype=bool)
                    for bx1, by1, bx2, by2 in confirmed:
                        m = (x >= bx1) & (x <= bx2) & (y >= by1) & (y <= by2)
                        is_pred[m] = True

                    # Pd: per (drone_id, win_idx) object
                    for did in np.unique(target_id[target_id > 0]):
                        dm = target_id == did
                        key = (int(did), win_idx)
                        obj_stats[key] = [int(dm.sum()), int((dm & is_pred).sum())]

                    # Fa: FP events → 2D CC blob count
                    fp_mask = (target_id == 0) & is_pred
                    fa_blobs += count_fa_blobs(x[fp_mask], y[fp_mask])
                    ev_frame_count += 1

                _p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                _r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                _f1 = 2 * _p * _r / (_p + _r) if (_p + _r) > 0 else 0.0
                pbar.set_postfix(P=f"{_p:.3f}", R=f"{_r:.3f}", F1=f"{_f1:.3f}",
                                 FP=total_fp, gt=gt_windows)

            if (vis or vis_only) and win_idx % cfg.vis_interval == 0:
                rgb_path = find_rgb(rgb_dir, win_idx, len(window_indices))
                out_path = os.path.join(vis_dir, f"win_{win_idx:05d}.png")
                visualize_v2(x, y, confirmed, gt_bboxes,
                             rgb_path, seq_id, win_idx, tp, fp, fn, out_path)

            if vid_proc is not None:
                frame = render_left_frame(x, y, confirmed, gt_bboxes, win_idx)
                vid_proc.stdin.write(frame.tobytes())

    if vid_proc is not None:
        vid_proc.stdin.close()
        vid_proc.wait()
        print(f"video: {vid_path}")

    if vis_only:
        return None

    # 잔여 tracklet 완료
    for frames in active_tracklets.values():
        completed_tracklets.append(frames)

    # Track-level evaluation
    trk_precision, drone_recall, trk_tp, trk_fp = evaluate_tracklets(
        completed_tracklets, drone_tracks, min_len=cfg.track_min_frames)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    coverage  = covered_windows / gt_windows if gt_windows > 0 else 0.0

    # event-level Pd/Fa
    detected = sum(1 for total, pred in obj_stats.values()
                   if total > 0 and pred / total >= EV_CORRECT_THRESH)
    ev_pd = detected / len(obj_stats) if obj_stats else 0.0
    ev_fa = fa_blobs / (ev_frame_count * SENSOR_H * SENSOR_W) if ev_frame_count > 0 else 0.0

    return {
        "seq":           seq_id,
        "precision":     precision,
        "recall":        recall,
        "f1":            f1,
        "coverage":      coverage,
        "tp":            total_tp,
        "fp":            total_fp,
        "fn":            total_fn,
        "gt_windows":    gt_windows,
        "total_windows": len(window_indices),
        "trk_precision": trk_precision,
        "drone_recall":  drone_recall,
        "trk_tp":        trk_tp,
        "trk_fp":        trk_fp,
        "ev_pd":         ev_pd,
        "ev_fa":         ev_fa,
        "ev_obj_total":  len(obj_stats),
        "ev_obj_detected": detected,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument("--seq",  type=int, default=None)
    parser.add_argument("--seqs", type=int, nargs="+", default=None)
    parser.add_argument("--vis",      action="store_true")
    parser.add_argument("--vis-only", action="store_true")
    parser.add_argument("--vod", action="store_true",
                        help="매 윈도우 좌 패널을 ffmpeg으로 mp4 저장 (평가 없음)")
    parser.add_argument("--rgb", action="store_true",
                        help="--vod와 함께 사용: 좌(RGB)+우(이벤트) 합성 영상")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--null-test", action="store_true",
                        help="GT bbox 위치 랜덤 셔플 → ev_pd null test")
    parser.add_argument("--save-video", action="store_true",
                        help="평가와 동시에 mp4 저장")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.out_dir = "log/detect_v2"
    os.makedirs(cfg.out_dir, exist_ok=True)

    if args.seq:
        seqs = [args.seq]
    elif args.seqs:
        seqs = args.seqs
    elif cfg.seqs:
        seqs = cfg.seqs
    else:
        split_dir = os.path.join(cfg.root, cfg.split)
        seqs = sorted(int(d) for d in os.listdir(split_dir) if d.isdigit())

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tags = []
    if cfg.buffer_win != 3:   tags.append(f"buf{cfg.buffer_win}")
    if cfg.adaptive_disp:     tags.append("adapdisp")
    elif cfg.disp_min != 5:   tags.append(f"disp{cfg.disp_min}")
    if cfg.alpha != 0.2:      tags.append(f"a{cfg.alpha}")
    run_tag = ("_" + "_".join(tags)) if tags else ""

    if args.vod:
        for seq_id in seqs:
            out_path = os.path.join(cfg.out_dir, f"{ts}_seq{seq_id}{run_tag}.mp4")
            run_sequence_to_video(seq_id, cfg, out_path, max_windows=args.max_windows, with_rgb=args.rgb)
        return

    results = []
    for seq_id in seqs:
        r = run_sequence(seq_id, cfg, vis=args.vis, vis_only=args.vis_only,
                         max_windows=args.max_windows, null_test=args.null_test,
                         save_video=args.save_video, vid_prefix=f"{ts}{run_tag}")
        if r:
            results.append(r)

    if not results:
        return

    # 출력
    print("\n" + "=" * 55)
    header = f"{'seq':>5} {'cov':>7} {'ev_pd':>7} {'ev_fa':>10} {'det/total':>12}"
    print(header)
    print("-" * 55)
    for r in results:
        print(f"{r['seq']:>5} {r['coverage']:>7.4f} {r['ev_pd']:>7.4f} "
              f"{r['ev_fa']:>10.6f} "
              f"{r['ev_obj_detected']:>5}/{r['ev_obj_total']:<5}")

    if len(results) > 1:
        det = sum(r['ev_obj_detected'] for r in results)
        tot = sum(r['ev_obj_total']    for r in results)
        avg_cov  = sum(r['coverage'] for r in results) / len(results)
        avg_evpd = det / tot if tot > 0 else 0.0
        avg_evfa = sum(r['ev_fa'] for r in results) / len(results)
        print("-" * 55)
        print(f"{'ALL':>5} {avg_cov:>7.4f} {avg_evpd:>7.4f} "
              f"{avg_evfa:>10.6f} {det:>5}/{tot:<5}")

    # CSV 저장
    csv_path = os.path.join(cfg.out_dir, f"{ts}{run_tag}_results.csv")
    fields = ["seq", "coverage", "ev_pd", "ev_fa", "ev_obj_total", "ev_obj_detected"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fields})
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
