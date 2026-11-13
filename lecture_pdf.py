"""Lecture video (YouTube URL or local file) -> notes PDF of board/slide snapshots.

  lecture-pdf "https://www.youtube.com/watch?v=..."            # full run
  lecture-pdf URL --sample                                     # download + 8 sample frames, then stop
  lecture-pdf URL --crop 0,0,1,0.955 --drop-pages 1,2          # rerun (no redownload / re-decode)

(or `python lecture_pdf.py ...` from a source checkout)

See README.md for the full workflow and tuning guide.
"""
import argparse
import glob
import re
import urllib.request
import json
import os
import shutil
import subprocess
import sys
import time

import cv2
import img2pdf
import numpy as np
from skimage.metrics import structural_similarity as ssim

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "selfie_multiclass_256x256.tflite"


def default_model_path():
    """Model next to the script (source checkout) if present, else a per-user cache (pip install)."""
    local = os.path.join(HERE, "models", MODEL_NAME)
    if os.path.exists(local):
        return local
    return os.path.join(os.path.expanduser("~"), ".cache", "lecture-pdf", MODEL_NAME)
SMALL_W, SMALL_H = 320, 180


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- decode
_HW = {}


def hw_pre(video):
    """ffmpeg input args for an NVIDIA (cuvid) decoder matching the video codec, or [] if unavailable."""
    if video not in _HW:
        _HW[video] = []
        try:
            codec = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                             "stream=codec_name", "-of", "csv=p=0", video], text=True).strip()
            dec = {"av1": "av1_cuvid", "h264": "h264_cuvid", "hevc": "hevc_cuvid", "vp9": "vp9_cuvid"}.get(codec)
            decs = subprocess.check_output(["ffmpeg", "-hide_banner", "-decoders"], text=True,
                                           stderr=subprocess.DEVNULL)
            if dec and dec in decs:
                _HW[video] = ["-c:v", dec]
        except (subprocess.CalledProcessError, OSError):
            pass
    return _HW[video]


def probe(video):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration", "-of", "json", video])
    j = json.loads(out)
    s = j["streams"][0]
    return s["width"], s["height"], float(j["format"]["duration"])


def decode_small(video, cache, gpu):
    """Decode at 1 fps, 320x180 gray, into a uint8 .npy (N,H,W)."""
    if os.path.exists(cache):
        return np.load(cache, mmap_mode="r")
    _, _, dur = probe(video)
    n_est = int(dur) + 2
    vf = f"fps=1,scale={SMALL_W}:{SMALL_H}:flags=area,format=gray"
    pre = (hw_pre(video) if gpu else []) or ["-threads", "4"]
    cmd = ["ffmpeg", "-v", "error", *pre, "-i", video, "-an", "-vf", vf,
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    log("decoding:", " ".join(cmd))
    buf = np.lib.format.open_memmap(cache + ".tmp.npy", mode="w+", dtype=np.uint8,
                                    shape=(n_est, SMALL_H, SMALL_W))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    fs = SMALL_W * SMALL_H
    n = 0
    while True:
        b = p.stdout.read(fs)
        if len(b) < fs:
            break
        if n < n_est:
            buf[n] = np.frombuffer(b, np.uint8).reshape(SMALL_H, SMALL_W)
        n += 1
        if n % 1800 == 0:
            log(f"  decoded {n//60} min")
    p.wait()
    if p.returncode != 0 or n == 0:
        del buf
        os.remove(cache + ".tmp.npy")
        if gpu and hw_pre(video):
            log("GPU decode failed, falling back to CPU (dav1d)")
            return decode_small(video, cache, gpu=False)
        sys.exit("decode failed")
    n = min(n, n_est)
    np.save(cache, np.asarray(buf[:n]))
    del buf
    os.remove(cache + ".tmp.npy")
    return np.load(cache, mmap_mode="r")


# ---------------------------------------------------------------- ink
def crop_px(crop, w, h):
    x0, y0, x1, y1 = crop
    return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)


def build_mask(crop, w, h, ignore):
    """Valid-pixel mask in cropped coordinates (ignore boxes are in full-frame fractions)."""
    cx0, cy0, cx1, cy1 = crop_px(crop, w, h)
    m = np.ones((cy1 - cy0, cx1 - cx0), np.uint8)
    for (a, b, c, d) in ignore:
        ax0, ay0, ax1, ay1 = crop_px((a, b, c, d), w, h)
        m[max(ay0 - cy0, 0):max(ay1 - cy0, 0), max(ax0 - cx0, 0):max(ax1 - cx0, 0)] = 0
    return m


def ink_mask(gray, light_on_dark, block, c):
    """Adaptive threshold: 1 where pixel is ink-colored."""
    if light_on_dark:
        return cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, block, -c)
    return cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, block, c)


def window_idx(t, window, spacing):
    """Causal window: frames t, t-s, t-2s, ... (window of them)."""
    return [t - k * spacing for k in range(window) if t - k * spacing >= 0]


# ---------------------------------------------------------------- detect
class Masks:
    """Bit-packed per-frame ink masks; masks[t] returns a uint8 0/1 array."""
    def __init__(self, packed, w):
        self.packed, self.w = packed, w

    def __getitem__(self, t):
        return np.unpackbits(self.packed[t], axis=1, count=self.w)


def detect(frames, a):
    n = len(frames)
    cx0, cy0, cx1, cy1 = crop_px(a.crop, SMALL_W, SMALL_H)
    fr = frames[:, cy0:cy1, cx0:cx1]
    valid = build_mask(a.crop, SMALL_W, SMALL_H, a.ignore)
    nvalid = valid.sum()
    blk = a.block_small
    kern = np.ones((3, 3), np.uint8)

    raw_ink = np.empty(n, np.float32)
    comp_ink = np.empty(n, np.float32)
    mh, mw = fr.shape[1:]
    packed = np.empty((n, mh, (mw + 7) // 8), np.uint8)  # composite ink masks, bit-packed
    masks = Masks(packed, mw)
    t0 = time.time()
    for t in range(n):
        raw_ink[t] = (ink_mask(np.ascontiguousarray(fr[t]), a.light_on_dark, blk, a.c) & valid).sum() / nvalid
        idx = window_idx(t, a.window, a.spacing)
        comp = np.median(fr[idx], axis=0).astype(np.uint8) if len(idx) > 1 else np.asarray(fr[t])
        m = ink_mask(comp, a.light_on_dark, blk, a.c) & valid
        packed[t] = np.packbits(m, axis=1)
        comp_ink[t] = m.sum() / nvalid
    log(f"composites: {n} frames in {time.time()-t0:.0f}s")

    # Page segmentation. Compare each composite with the one `lag` s earlier, where
    # lag = window span + 1, so the two composites share no frames: writing only ever
    # adds ink, while an erase or a slide change makes most of the old ink vanish.
    span = (a.window - 1) * a.spacing
    lag = span + 1
    # A new slide can also arrive without removing ink (e.g. break screen over the board),
    # so also track the fraction of current ink that is new.
    lost = np.zeros(n, np.float32)
    gained = np.zeros(n, np.float32)
    for t in range(lag, n):
        prev, cur = masks[t - lag], masks[t]
        ps, cs = prev.sum(), cur.sum()
        if ps > a.min_ink * nvalid:
            lost[t] = (prev & (1 - cv2.dilate(cur, kern))).sum() / ps
        if cs > a.min_ink * nvalid:
            gained[t] = (cur & (1 - cv2.dilate(prev, kern))).sum() / cs

    cuts = []  # (page start, cut index, reason); at the cut the composite already shows the new page
    start, peak = 0, comp_ink[0]
    t = 1
    while t < n:
        hold = slice(t, min(n, t + a.sustain))
        changed = bool(np.all(lost[hold] > a.change_ratio)) or bool(np.all(gained[hold] > a.gain_ratio))
        drop = peak > a.min_ink and bool(np.all(comp_ink[hold] < (1 - a.drop_ratio) * peak))
        if (changed or drop) and t - start >= a.min_page:
            cuts.append((start, t, f"change(lost{lost[t]:.2f},new{gained[t]:.2f})" if changed else f"drop{1-comp_ink[t]/peak:.2f}"))
            start = t
            t += lag  # wait until the composites no longer mix old and new page
            peak = comp_ink[min(t, n - 1)]
            continue
        peak = max(peak, comp_ink[t])
        t += 1
    cuts.append((start, n, "end"))

    pages = []
    for (s, d, why) in cuts:
        if d >= n:  # last segment: fullest composite
            best = s + int(np.argmax(comp_ink[s:d]))
        else:
            # The composite flips ~span/2 s after the real clear T; find T as the
            # sharpest drop in the raw (un-composited) ink series just before d.
            lo = max(s + 1, d - span - 3)
            best_T, best_score = d, -1e9
            for T in range(lo, d + 1):
                before = np.median(raw_ink[max(s, T - 3):T])
                after = np.median(raw_ink[T:min(n, T + 3)])
                if before - after > best_score:
                    best_score, best_T = before - after, T
            best = max(s, best_T - 1)  # composite of frames strictly before the clear
            # if an earlier composite in the page is clearly fuller, prefer it
            m_i = s + int(np.argmax(comp_ink[s:best + 1]))
            if comp_ink[m_i] > comp_ink[best] * 1.15:
                best = m_i
        pages.append(dict(start=int(s), end=int(d), t=int(best), ink=float(comp_ink[best]), why=why))
    return pages, masks, comp_ink, raw_ink


# ---------------------------------------------------------------- dedupe
def dedupe(pages, frames, masks, a):
    cx0, cy0, cx1, cy1 = crop_px(a.crop, SMALL_W, SMALL_H)
    fr = frames[:, cy0:cy1, cx0:cx1]
    kern = np.ones((5, 5), np.uint8)
    kept = []
    for p in pages:
        if p["ink"] < a.min_ink:
            log(f"  drop empty   t={mmss(p['t'])} ink={p['ink']:.4f}")
            continue
        p["comp"] = np.median(fr[window_idx(p["t"], a.window, a.spacing)], axis=0).astype(np.uint8)
        if kept:
            q = kept[-1]
            mp, mq = masks[p["t"]], masks[q["t"]]
            s = ssim(p["comp"], q["comp"], data_range=255)
            # containment: how much of one page's ink appears in the other
            q_in_p = (mq & cv2.dilate(mp, kern)).sum() / max(mq.sum(), 1)
            p_in_q = (mp & cv2.dilate(mq, kern)).sum() / max(mp.sum(), 1)
            if s > a.ssim_dup or (q_in_p > a.contain and p_in_q > a.contain):
                # near-identical: keep the fuller one
                log(f"  dup          {mmss(q['t'])} ~ {mmss(p['t'])} ssim={s:.3f}")
                if p["ink"] > q["ink"]:
                    kept[-1] = p
                continue
            if q_in_p > a.contain:  # previous page is a subset of this one
                log(f"  subset       {mmss(q['t'])} within {mmss(p['t'])}")
                kept[-1] = p
                continue
            if p_in_q > a.contain:  # this page is a subset of previous one
                log(f"  subset       {mmss(p['t'])} within {mmss(q['t'])}")
                continue
        kept.append(p)
    return kept


# ---------------------------------------------------------------- export
def grab_full(video, t, w, h, gpu):
    pre = hw_pre(video) if gpu else []
    cmd = ["ffmpeg", "-v", "error", *pre, "-ss", f"{t}", "-i", video, "-frames:v", "1",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    b = subprocess.run(cmd, capture_output=True).stdout
    if len(b) < w * h * 3:
        return grab_full(video, t, w, h, False) if pre else None
    return np.frombuffer(b[:w * h * 3], np.uint8).reshape(h, w, 3)


def mmss(t):
    t = int(t)
    return f"{t//3600}:{t//60%60:02d}:{t%60:02d}" if t >= 3600 else f"{t//60:02d}:{t%60:02d}"


def grab_clip(video, t0, n, spacing, w, h, gpu):
    """n full-res frames at t0, t0+spacing, ... from one sequential decode."""
    pre = hw_pre(video) if gpu else []
    cmd = ["ffmpeg", "-v", "error", *pre, "-ss", f"{max(t0, 0)}", "-i", video, "-t", f"{(n - 1) * spacing + 0.5}",
           "-an", "-vf", f"fps=1/{spacing}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    b = subprocess.run(cmd, capture_output=True).stdout
    k = len(b) // (w * h * 3)
    if k == 0 and pre:
        return grab_clip(video, t0, n, spacing, w, h, False)
    return np.frombuffer(b[:k * w * h * 3], np.uint8).reshape(k, h, w, 3)


class PersonMasker:
    """MediaPipe multiclass selfie segmenter -> dilated 'person' mask at full res."""
    def __init__(self, model, dilate_px, thresh):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self.mp = mp
        opts = vision.ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=model),
                                            output_confidence_masks=True, output_category_mask=False)
        self.seg = vision.ImageSegmenter.create_from_options(opts)
        self.kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        self.thresh = thresh

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        small = cv2.cvtColor(cv2.resize(bgr, (640, int(640 * h / w)), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        res = self.seg.segment(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=small))
        person = 1 - np.squeeze(res.confidence_masks[0].numpy_view())  # class 0 = background
        m = cv2.resize((person > self.thresh).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.dilate(m, self.kern).astype(bool)


def unoccluded_composite(frames, masks, latest_k):
    """Per pixel: median of the latest `latest_k` frames where the person isn't covering it.
    Pixels covered in every frame fall back to the plain temporal median."""
    frames, masks = frames[::-1], masks[::-1]  # newest first
    free = ~masks
    rank = np.cumsum(free, axis=0)
    use = free & (rank <= latest_k)
    f = frames.astype(np.float32)
    f[~use] = np.nan
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = np.nanmedian(f, axis=0)
    never = ~free.any(axis=0)
    out[never] = np.median(frames[:, never], axis=0)
    return np.nan_to_num(out).astype(np.uint8), float(never.mean())


def export(kept, a):
    W, H, _ = probe(a.video)
    if a.person_mask:
        ensure_model(a.mask_model)
    out = os.path.join(a.out, "pages")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    x0, y0, x1, y1 = crop_px(a.crop, W, H)
    masker = PersonMasker(a.mask_model, a.mask_dilate, a.mask_thresh) if a.person_mask else None
    span = (a.window - 1) * a.spacing
    paths, lines = [], []
    for i, p in enumerate(kept, 1):
        if masker:
            # frames from the page's own lifetime only, ending at the snapshot time
            lo = max(0, p["start"] - span // 2)
            n = max(1, min(a.mask_frames, (p["t"] - lo) // a.mask_spacing + 1))
            fr = grab_clip(a.video, p["t"] - (n - 1) * a.mask_spacing, n, a.mask_spacing, W, H, a.gpu)
            ms = np.stack([masker(f) for f in fr])
            img, hidden = unoccluded_composite(fr[:, y0:y1, x0:x1], ms[:, y0:y1, x0:x1], a.mask_latest)
            note = f"{len(fr)} frames, {hidden*100:.1f}% never visible"
        else:
            ims = [grab_full(a.video, tt, W, H, a.gpu) for tt in window_idx(p["t"], a.window, a.spacing)]
            ims = [im[y0:y1, x0:x1] for im in ims if im is not None]
            img = np.median(np.stack(ims), axis=0).astype(np.uint8) if len(ims) > 1 else ims[0]
            note = f"median of {len(ims)}"
        path = os.path.join(out, f"page_{i:03d}.png")
        cv2.imwrite(path, img)
        paths.append(path)
        lines.append(f"{i:3d}  {mmss(p['t'])}   (page visible {mmss(p['start'])}-{mmss(p['end'])}, cut: {p['why']})")
        log(f"  page {i:3d} @ {mmss(p['t'])}  {note}")
    if masker:
        masker.seg.close()
    with open(os.path.join(a.out, "notes.pdf"), "wb") as f:
        f.write(img2pdf.convert(paths))
    with open(os.path.join(a.out, "timestamps.txt"), "w") as f:
        f.write("page  snapshot  (mm:ss or h:mm:ss into the video)\n")
        f.write("\n".join(lines) + "\n")
    contact_sheet(paths, kept, os.path.join(a.out, "contact_sheet.jpg"))
    return paths


def contact_sheet(paths, kept, dst, cols=8, tw=300):
    thumbs = []
    for i, (pth, p) in enumerate(zip(paths, kept), 1):
        im = cv2.imread(pth)
        th = cv2.resize(im, (tw, int(im.shape[0] * tw / im.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.rectangle(th, (0, 0), (tw, 22), (0, 0, 0), -1)
        cv2.putText(th, f"{i}  {mmss(p['t'])}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        thumbs.append(cv2.copyMakeBorder(th, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    h, w = thumbs[0].shape[:2]
    while len(thumbs) % cols:
        thumbs.append(np.full((h, w, 3), 255, np.uint8))
    rows = [np.hstack(thumbs[r:r + cols]) for r in range(0, len(thumbs), cols)]
    cv2.imwrite(dst, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])




# ---------------------------------------------------------------- source / setup
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
             "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite")
LINK_RE = re.compile(r"pdf|notes|dpp|drive\.google|docs\.google|telegram|t\.me/|dropbox|mega\.nz|worksheet|handout",
                     re.I)


def is_url(s):
    return bool(re.match(r"https?://", s))


def ytdlp(*args, capture=False):
    cmd = [sys.executable, "-m", "yt_dlp", *args]
    if capture:
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8").stdout
    return subprocess.run(cmd).returncode


def fetch_info(url, out):
    """Video metadata via yt-dlp (cached to out/download/info.json)."""
    path = os.path.join(out, "download", "info.json")
    if not os.path.exists(path):
        raw = ytdlp("-J", "--skip-download", "--no-warnings", url, capture=True)
        if not raw.strip():
            sys.exit(f"yt-dlp could not read {url}")
        with open(path, "w", encoding="utf-8") as f:
            f.write(raw)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def scan_for_notes(info, url, out, comments):
    """Lines from the description (and optionally top comments) that look like notes/PDF links."""
    hits = [ln.strip() for ln in (info.get("description") or "").splitlines() if LINK_RE.search(ln)]
    if comments:
        cpath = os.path.join(out, "download", "comments.info.json")
        if not os.path.exists(cpath):
            ytdlp("--skip-download", "--write-comments", "--no-warnings", "--extractor-args",
                  f"youtube:max_comments={comments},all,0,0;comment_sort=top",
                  "-o", os.path.join(out, "download", "comments"), url, capture=True)
        if os.path.exists(cpath):
            with open(cpath, encoding="utf-8") as f:
                for c in json.load(f).get("comments") or []:
                    t = c.get("text", "")
                    if LINK_RE.search(t) and "http" in t:
                        hits.append(f"[comment by {c.get('author')}] {t[:200]}".replace("\n", " "))
    return hits


def _downloaded(dl):
    return [p for p in glob.glob(os.path.join(dl, "video.*")) if not p.endswith((".part", ".ytdl"))]


def download(url, out, max_height):
    dl = os.path.join(out, "download")
    have = _downloaded(dl)
    if have:
        return have[0]
    fmt = f"bv*[height<={max_height}][ext=mp4]/bv*[height<={max_height}]/b[height<={max_height}]"
    log(f"downloading (video only, <= {max_height}p) ...")
    if ytdlp("-f", fmt, "--no-progress", "-o", os.path.join(dl, "video.%(ext)s"), url) != 0:
        sys.exit("download failed")
    have = _downloaded(dl)
    if not have:
        sys.exit("download produced no file")
    return have[0]


def ensure_model(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        log(f"downloading person-segmentation model -> {path}")
        urllib.request.urlretrieve(MODEL_URL, path)


def sample_montage(video, out, n=8):
    """n gridded sample frames (skipping first/last 5%) for choosing --crop and ink polarity."""
    _, _, dur = probe(video)
    d = os.path.join(out, "samples")
    os.makedirs(d, exist_ok=True)
    ts = [int(dur * (0.05 + 0.9 * i / (n - 1))) for i in range(n)]
    tiles = []
    for t in ts:
        b = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", video, "-frames:v", "1",
                            "-vf", "scale=640:360", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                           capture_output=True).stdout
        im = (np.frombuffer(b[:640 * 360 * 3], np.uint8).reshape(360, 640, 3).copy()
              if len(b) >= 640 * 360 * 3 else np.zeros((360, 640, 3), np.uint8))
        # grid every 10% so crop fractions can be read off directly
        for k in range(1, 10):
            cv2.line(im, (64 * k, 0), (64 * k, 359), (0, 255, 255), 1)
            cv2.line(im, (0, 36 * k), (639, 36 * k), (0, 255, 255), 1)
            cv2.putText(im, f".{k}", (64 * k + 2, 12), 0, 0.35, (0, 255, 255), 1)
            cv2.putText(im, f".{k}", (2, 36 * k - 2), 0, 0.35, (0, 255, 255), 1)
        cv2.putText(im, mmss(t), (540, 350), 0, 0.6, (0, 0, 255), 2)
        tiles.append(im)
    grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
    path = os.path.join(d, "montage.jpg")
    cv2.imwrite(path, grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


def video_id(src):
    m = re.search(r"(?:v=|youtu\.be/|shorts/|live/)([\w-]{11})", src)
    if m:
        return m.group(1)
    return re.sub(r"\W+", "_", os.path.splitext(os.path.basename(src))[0])[:60] or "video"


# ---------------------------------------------------------------- main
def floats(s):
    v = tuple(float(x) for x in s.split(","))
    if len(v) != 4:
        raise argparse.ArgumentTypeError("expected 4 comma-separated fractions x0,y0,x1,y1")
    return v


def ints(s):
    return {int(x) for x in s.split(",") if x.strip()}


def main():
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    ap = argparse.ArgumentParser(prog="lecture-pdf", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="YouTube (or any yt-dlp supported) URL, or a local video file")
    ap.add_argument("--out", help="output folder (default: ./lecture_notes/<video id>)")
    g = ap.add_argument_group("source")
    g.add_argument("--max-height", type=int, default=720, help="download resolution cap")
    g.add_argument("--comments", type=int, default=0, help="also scan this many top comments for notes links")
    g.add_argument("--stop-if-notes", action="store_true",
                   help="exit (code 3) before downloading if the description/comments contain a notes/PDF link")
    g.add_argument("--sample", action="store_true", help="only download + write samples/montage.jpg, then stop")
    g = ap.add_argument_group("board region / ink")
    g.add_argument("--crop", type=floats, default=(0, 0, 1, 0.955),
                   help="board region as fractions x0,y0,x1,y1 (default: full width, drop bottom 4.5%%)")
    g.add_argument("--ignore", type=floats, action="append",
                   help="x0,y0,x1,y1 fraction box to ignore for ink, e.g. a logo/clock (repeatable; "
                        "default 0.93,0,1,0.11 = top-right corner; pass 0,0,0,0 for none)")
    g.add_argument("--dark-on-light", dest="light_on_dark", action="store_false",
                   help="ink is dark on a light board/slide (default: light ink on dark board)")
    g.add_argument("--c", type=float, default=20, help="adaptive threshold offset (higher = less noise)")
    g.add_argument("--block-small", type=int, default=15, help="adaptive threshold block (320px frames)")
    g = ap.add_argument_group("page detection")
    g.add_argument("--window", type=int, default=7, help="frames in the median window")
    g.add_argument("--spacing", type=int, default=3, help="seconds between median window frames")
    g.add_argument("--drop-ratio", type=float, default=0.45,
                   help="cut when ink falls this fraction below the page's running peak")
    g.add_argument("--change-ratio", type=float, default=0.5,
                   help="cut when this fraction of ink from one window-span earlier has vanished (erase/slide change)")
    g.add_argument("--gain-ratio", type=float, default=0.6,
                   help="cut when this fraction of current ink is new vs one window-span earlier (slide replaced)")
    g.add_argument("--sustain", type=int, default=5, help="seconds a cut condition must hold (rejects noise)")
    g.add_argument("--min-page", type=int, default=20, help="min seconds per page")
    g = ap.add_argument_group("dedupe / filtering")
    g.add_argument("--min-ink", type=float, default=0.025, help="pages with less ink fraction are dropped")
    g.add_argument("--ssim-dup", type=float, default=0.92, help="SSIM above this vs previous page = duplicate")
    g.add_argument("--contain", type=float, default=0.9, help="ink containment above this = subset page")
    g.add_argument("--drop-pages", type=ints, default=set(),
                   help="comma list of page numbers (as numbered in the previous contact sheet) to leave out")
    g = ap.add_argument_group("instructor removal (export)")
    g.add_argument("--no-person-mask", dest="person_mask", action="store_false",
                   help="plain median compositing instead of MediaPipe person masking")
    g.add_argument("--mask-model", default=default_model_path(), help="MediaPipe selfie-multiclass model (auto-downloaded)")
    g.add_argument("--mask-frames", type=int, default=20, help="frames per page for person-mask compositing")
    g.add_argument("--mask-spacing", type=int, default=2, help="seconds between those frames")
    g.add_argument("--mask-dilate", type=int, default=24, help="grow person mask by this many px (720p)")
    g.add_argument("--mask-thresh", type=float, default=0.3, help="person confidence threshold")
    g.add_argument("--mask-latest", type=int, default=3, help="median of this many latest unoccluded samples")
    g = ap.add_argument_group("misc")
    g.add_argument("--no-gpu", dest="gpu", action="store_false", help="never use the NVIDIA hardware decoder")
    g.add_argument("--detect-only", action="store_true", help="print the page list, skip export")
    a = ap.parse_args()
    if a.ignore is None:
        a.ignore = [(0.93, 0.0, 1.0, 0.11)]
    a.ignore = [b for b in a.ignore if b[2] > b[0] and b[3] > b[1]]

    T0 = time.time()
    a.out = os.path.abspath(a.out or os.path.join("lecture_notes", video_id(a.source)))
    os.makedirs(os.path.join(a.out, "download"), exist_ok=True)
    log(f"output folder: {a.out}")

    if is_url(a.source):
        info = fetch_info(a.source, a.out)
        log(f"title: {info.get('title')}  ({info.get('duration_string')})")
        hits = scan_for_notes(info, a.source, a.out, a.comments)
        if hits:
            log("possible notes/PDF links found:")
            for h in hits:
                log("   ", h)
            if a.stop_if_notes:
                log("stopping (--stop-if-notes)")
                sys.exit(3)
        else:
            log("no notes/PDF links in description" + (" or comments" if a.comments else ""))
        if info.get("chapters"):
            with open(os.path.join(a.out, "chapters.txt"), "w", encoding="utf-8") as f:
                f.writelines(f"{mmss(c['start_time'])}  {c['title']}\n" for c in info["chapters"])
        a.video = download(a.source, a.out, a.max_height)
    else:
        if not os.path.exists(a.source):
            sys.exit(f"no such file: {a.source}")
        a.video = os.path.abspath(a.source)

    W, H, dur = probe(a.video)
    dec = (hw_pre(a.video) or ["", "none (CPU)"])[1] if a.gpu else "off (CPU)"
    log(f"video: {W}x{H}, {mmss(dur)}, hw decoder: {dec}")
    montage = os.path.join(a.out, "samples", "montage.jpg")
    if a.sample or not os.path.exists(montage):
        log(f"sample montage: {sample_montage(a.video, a.out)}")
    if a.sample:
        return

    frames = decode_small(a.video, os.path.join(a.out, "frames_320.npy"), a.gpu)
    log(f"frames: {frames.shape}  ({time.time()-T0:.0f}s)")
    log(f"settings: crop={a.crop} ignore={a.ignore} light_on_dark={a.light_on_dark} "
        f"window={a.window}x{a.spacing}s drop={a.drop_ratio} change={a.change_ratio} gain={a.gain_ratio}")
    pages, masks, comp_ink, raw_ink = detect(frames, a)
    log(f"segments: {len(pages)}")
    kept = dedupe(pages, frames, masks, a)
    if a.drop_pages:
        kept = [p for i, p in enumerate(kept, 1) if i not in a.drop_pages]
        log(f"dropped pages {sorted(a.drop_pages)} on request")
    log(f"kept pages: {len(kept)}  ({time.time()-T0:.0f}s)")
    if a.detect_only:
        for i, p in enumerate(kept, 1):
            log(f"{i:3d} {mmss(p['t'])} ink={p['ink']:.4f} {p['why']}")
        return
    if not kept:
        sys.exit("no pages found - check --crop / --dark-on-light against samples/montage.jpg")
    export(kept, a)
    run = {k: (sorted(v) if isinstance(v, set) else v) for k, v in vars(a).items()}
    run.update(pages=len(kept), runtime_s=round(time.time() - T0))
    with open(os.path.join(a.out, "run.json"), "w") as f:
        json.dump(run, f, indent=1, default=str)
    log(f"done: {len(kept)} pages, total runtime {time.time()-T0:.0f}s")
    log(f"  pdf:        {os.path.join(a.out, 'notes.pdf')}")
    log(f"  timestamps: {os.path.join(a.out, 'timestamps.txt')}")
    log(f"  contact:    {os.path.join(a.out, 'contact_sheet.jpg')}")


if __name__ == "__main__":
    main()
