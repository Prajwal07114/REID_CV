"""
main.py
───────
Person ReID & Multi-Camera Tracking System  ·  v3.0

Entry point. Parses CLI args, constructs all modules, and runs the
multi-camera processing loop.

Usage:
    python main.py --cams cam1.mp4 cam2.mp4 cam3.mp4
    python main.py --cams cam1.mp4 cam2.mp4 --model osnet --threshold 0.55
    python main.py --cams cam1.mp4 --eval --eval-root /data/Market-1501
    python main.py --cams cam1.mp4 cam2.mp4 --ablation
    python main.py --sanity-check
"""

import argparse
import time
import warnings

import torch

from config.settings       import SystemConfig
from detection.detector    import PersonDetector
from reid.gallery          import GalleryManager
from reid.pipeline         import ReIDPipeline, build_engine
from evaluation.market1501 import ReIDEvaluator
from evaluation.metrics    import print_results, save_results
from benchmarking.ablation import AblationRunner
from visualization.hud     import CameraProcessor
from utils.io              import validate_video_paths, save_session
from utils.logging_utils   import setup_logging
from utils.sanity_check    import run_embedding_sanity_check

warnings.filterwarnings("ignore")
setup_logging()


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Person ReID Multi-Camera Tracking System v3"
    )
    ap.add_argument("--cams",         nargs="+",  default=["cam1.mp4", "cam2.mp4"])
    ap.add_argument("--width",        type=int,   default=960)
    ap.add_argument("--threshold",    type=float, default=0.55)
    ap.add_argument("--model",        type=str,   default="osnet",
                    choices=["osnet", "resnet50", "resnet18"])
    ap.add_argument("--tracker",      type=str,   default="bytetrack",
                    choices=["bytetrack", "iou"])
    ap.add_argument("--det-model",    type=str,   default="yolov8m.pt")
    ap.add_argument("--eval",         action="store_true")
    ap.add_argument("--eval-dataset", type=str,   default="market1501",
                    choices=["market1501", "dukemtmc"])
    ap.add_argument("--eval-root",    type=str,   default="")
    ap.add_argument("--ablation",     action="store_true")
    ap.add_argument("--models",       nargs="+",
                    default=["resnet18", "resnet50", "osnet"])
    ap.add_argument("--trackers",     nargs="+",
                    default=["iou", "bytetrack"])
    ap.add_argument("--sanity-check", action="store_true",
                    help="Run embedding sanity check and exit")
    return ap


def _build_cfg(args: argparse.Namespace) -> SystemConfig:
    """Build an immutable config from parsed CLI args. No global mutation."""
    return SystemConfig(
        proc_width    = args.width,
        reid_model    = args.model,
        reid_threshold= args.threshold,
        tracker_type  = args.tracker,
        det_model     = args.det_model,
        eval_dataset  = args.eval_dataset,
        eval_root     = args.eval_root,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = _build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = _build_cfg(args)

    # ── Sanity check mode ──────────────────────────────────────────────────
    if args.sanity_check:
        engine = build_engine(cfg.reid_model, device)
        run_embedding_sanity_check(engine)
        return

    # ── Validate inputs ────────────────────────────────────────────────────
    if not validate_video_paths(args.cams):
        return

    print("\n" + "=" * 68)
    print("  Person ReID & Multi-Camera Tracking System  ·  v3.0")
    print("=" * 68)
    print(f"  Cameras   : {args.cams}")
    print(f"  ReID      : {args.model.upper()}")
    print(f"  Tracker   : {args.tracker.upper()}")
    print(f"  Threshold : {args.threshold}")
    print(f"  Device    : {device}")
    print("=" * 68 + "\n")

    # ── Ablation mode ──────────────────────────────────────────────────────
    if args.ablation:
        AblationRunner(
            args.cams, args.models, args.trackers, device, cfg
        ).run()
        return

    # ── Standard processing ────────────────────────────────────────────────
    engine   = build_engine(args.model, device)
    gallery  = GalleryManager(cfg)
    pipeline = ReIDPipeline(engine, gallery)
    detector = PersonDetector(cfg, device)

    processors = []
    for i, path in enumerate(args.cams):
        try:
            processors.append(
                CameraProcessor(
                    cam_id=f"cam{i + 1}",
                    video_path=path,
                    tracker_type=args.tracker,
                    reid_pipeline=pipeline,
                    cfg=cfg,
                )
            )
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            return

    max_frames = max(p.total for p in processors)
    t0         = time.time()
    print(f"[INFO] Processing {max_frames} frames × {len(processors)} cameras...\n")

    for fidx in range(max_frames):
        # FIX: batched detection across all cameras in one YOLO forward pass
        frames = [p.read_frame() for p in processors]
        if not any(f is not None for f in frames):
            break

        valid      = [(i, f) for i, f in enumerate(frames) if f is not None]
        batch_dets = detector.detect_batch([f for _, f in valid])

        for (i, frame), dets in zip(valid, batch_dets):
            processors[i].step(frame, dets)

        if fidx % 15 == 0:
            elapsed  = time.time() - t0
            pct      = (fidx + 1) / max(max_frames, 1) * 100
            bar      = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            fps_proc = (fidx + 1) / max(elapsed, 0.01)
            total_sw = sum(p.id_switches for p in processors)
            print(
                f"\r  [{bar}] {pct:5.1f}%  frame {fidx+1:>5}/{max_frames}"
                f"  {fps_proc:.1f} fps  GIDs:{gallery.num_ids:>3}"
                f"  ID-SW:{total_sw}",
                end="", flush=True,
            )

    elapsed = time.time() - t0
    print(f"\n\n{'='*68}")
    print(f"  DONE in {elapsed:.1f}s  ({max_frames/elapsed:.1f} fps avg)")
    print(f"{'='*68}")

    summaries = []
    for proc in processors:
        proc.close()
        s = proc.summary()
        summaries.append(s)
        print(
            f"  [{s['cam_id']}] entries={s['entries']}  "
            f"id_switches={s['id_switches']}  → {s['output']}"
        )

    print(f"\n  Total global identities : {gallery.num_ids}")
    print(f"  Total ID switches       : {sum(s['id_switches'] for s in summaries)}")
    print(f"{'='*68}\n")

    session = {
        "version":    "v3.0",
        "model":      args.model,
        "tracker":    args.tracker,
        "threshold":  args.threshold,
        "global_ids": gallery.num_ids,
        "cameras":    summaries,
    }
    save_session(session)

    # ── Optional evaluation ────────────────────────────────────────────────
    if args.eval:
        if not args.eval_root:
            print("[WARNING] --eval-root not specified. Skipping evaluation.")
        else:
            evaluator = ReIDEvaluator(engine, args.eval_root, args.eval_dataset)
            results   = evaluator.run()
            print_results(results, model=args.model)
            save_results(results, model=args.model)


if __name__ == "__main__":
    main()
