# Person Re-ID & Multi-Camera Tracking System · v3.0

A production-style, modular implementation of a person re-identification and multi-camera tracking pipeline.

**Stack:** YOLOv8 → ByteTrack + Kalman → OSNet → Cross-camera cosine matching → Market-1501 evaluation

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         main.py  (orchestrator)                  │
└───────────┬───────────────┬──────────────────┬───────────────────┘
            │               │                  │
    ┌───────▼──────┐  ┌─────▼──────┐  ┌───────▼───────┐
    │  Detection   │  │  Tracking  │  │     ReID      │
    │  (YOLO v8)   │  │            │  │               │
    │              │  │ ┌────────┐ │  │ ┌───────────┐ │
    │ detect()     │  │ │Kalman  │ │  │ │EmbEngine  │ │
    │ detect_batch │  │ │[x,y,a,h│ │  │ │ (OSNet /  │ │
    └──────────────┘  │ └────────┘ │  │ │  ResNet)  │ │
                      │ ┌────────┐ │  │ └───────────┘ │
                      │ │Tracklet│ │  │ ┌───────────┐ │
                      │ └────────┘ │  │ │  Gallery  │ │
                      │ ┌────────┐ │  │ │ (EMA cos) │ │
                      │ │Byte-   │ │  │ └───────────┘ │
                      │ │Track   │ │  │ ┌───────────┐ │
                      │ └────────┘ │  │ │ Pipeline  │ │
                      └────────────┘  │ └───────────┘ │
                                      └───────────────┘
            │               │                  │
    ┌───────▼───────────────▼──────────────────▼───────┐
    │              Visualization (HUD)                  │
    │   heatmap · trajectory · entry/exit counter       │
    └───────────────────────────────────────────────────┘
            │
    ┌───────▼──────────────────────────────────────────┐
    │              Evaluation & Benchmarking            │
    │   Market-1501 mAP/Rank-N · Ablation grid search  │
    └──────────────────────────────────────────────────┘
```

---

## Repository Structure

```
reid-multicam-v3/
│
├── main.py                    # CLI entry point
├── requirements.txt
├── README.md
│
├── config/
│   └── settings.py            # Immutable SystemConfig dataclass
│
├── detection/
│   └── detector.py            # YOLOv8 wrapper; detect() / detect_batch()
│
├── tracking/
│   ├── kalman.py              # [x,y,a,h] Kalman filter (SORT/ByteTrack standard)
│   ├── tracker.py             # ByteTracker + IoUTracker
│   └── tracklet.py            # Tracklet dataclass
│
├── reid/
│   ├── embedding_engine.py    # Abstract EmbeddingEngine base class
│   ├── osnet_engine.py        # OSNet (Market-1501 ReID-pretrained)
│   ├── resnet_engine.py       # ResNet baseline (ImageNet-pretrained)
│   ├── gallery.py             # Cross-camera identity gallery (EMA matching)
│   └── pipeline.py            # ReIDPipeline; build_engine() factory
│
├── evaluation/
│   ├── market1501.py          # Correct Market-1501 protocol
│   └── metrics.py             # print_results / save_results
│
├── benchmarking/
│   └── ablation.py            # Tracker × backbone grid search
│
├── visualization/
│   └── hud.py                 # Drawing utils + CameraProcessor
│
├── utils/
│   ├── io.py                  # Session I/O helpers
│   ├── geometry.py            # BBox geometry utilities
│   ├── logging_utils.py       # Logging configuration
│   └── sanity_check.py        # Embedding engine verification
│
└── outputs/                   # Written by the pipeline at runtime
```

---

## Key Fixes (v2 → v3)

| # | Fix | Impact |
|---|-----|--------|
| 1 | **OSNet loading** — `num_classes=1` + strip classifier; Market-1501 pretrained weights | Correct embedding space for cosine matching |
| 2 | **Market-1501 distractor handling** — keep `pid==-1` in gallery at load time | Results comparable to published benchmarks |
| 3 | **ByteTrack double-increment** — newly-lost tracks no longer incremented twice | Correct occlusion lifetime; fewer ID switches |
| 4 | **Kalman state space** — `[x,y,a,h]` (aspect ratio) instead of `[cx,cy,w,h]` | Scale-invariant, numerically stable filter |
| – | **detect_batch** wired into the main loop | One YOLO call for all cameras per frame |
| – | **Gallery buffer** uses `deque(maxlen=)` not `list.pop(0)` | O(1) instead of O(N) append |
| – | **Global config** never mutated; passed via constructor | Thread-safe; testable in isolation |

---

## Installation

```bash
git clone https://github.com/yourname/reid-multicam-v3
cd reid-multicam-v3
pip install -r requirements.txt
```

> **torchreid note:** if `pip install torchreid` fails, install from source:
> ```bash
> git clone https://github.com/KaiyangZhou/deep-person-reid
> cd deep-person-reid && pip install -e .
> ```

---

## Usage

### Standard multi-camera run
```bash
python main.py --cams cam1.mp4 cam2.mp4 cam3.mp4
```

### Select model and threshold
```bash
python main.py --cams cam1.mp4 cam2.mp4 --model osnet --threshold 0.55
```

### Market-1501 evaluation
```bash
python main.py --cams cam1.mp4 --eval --eval-root /data/Market-1501
```

### Ablation study (tracker × backbone grid)
```bash
python main.py --cams cam1.mp4 cam2.mp4 --ablation \
    --models resnet18 resnet50 osnet \
    --trackers iou bytetrack
```

### Sanity check (verify embeddings are valid)
```bash
python main.py --sanity-check --model osnet
```

---

## Benchmark Results

> Run `--ablation` to populate this table with your hardware's numbers.

| Backbone | Tracker    | FPS | Global IDs | ID Switches |
|----------|------------|-----|------------|-------------|
| ResNet18 | IoU        | —   | —          | —           |
| ResNet18 | ByteTrack  | —   | —          | —           |
| ResNet50 | IoU        | —   | —          | —           |
| ResNet50 | ByteTrack  | —   | —          | —           |
| OSNet    | IoU        | —   | —          | —           |
| OSNet    | ByteTrack  | —   | —          | —           |

---

## Evaluation (Market-1501)

1. Download [Market-1501](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html).
2. Unzip so the directory contains `query/` and `bounding_box_test/`.
3. Run:
   ```bash
   python main.py --cams cam1.mp4 --eval \
       --eval-root /path/to/Market-1501 \
       --model osnet
   ```

Expected reference numbers (OSNet-x1_0, Market-1501 pretrained via torchreid):

| mAP | Rank-1 | Rank-5 | Rank-10 |
|-----|--------|--------|---------|
| ~74% | ~94% | ~98% | ~99% |

---

## Interview-Ready Explanation

**What does this system do?**
Given N video streams from different cameras covering the same area, the system detects every person in every frame, assigns them a stable camera-local track ID using ByteTrack, extracts a 512-dimensional identity embedding via OSNet, and then matches those embeddings across cameras using cosine similarity to assign a single global ID per physical person.

**Why OSNet over ResNet for ReID?**
OSNet is trained with triplet loss on ReID datasets (Market-1501), which explicitly optimises the embedding space so that cosine similarity is high for same-identity pairs and low for different-identity pairs. An ImageNet ResNet backbone is trained with cross-entropy loss for category classification — it produces embeddings that cluster by appearance category (jacket colour, clothing type), not by individual identity. Using cosine similarity on raw ImageNet features gives near-random matching performance.

**Why [x,y,a,h] in the Kalman filter?**
Using aspect ratio `a = w/h` instead of raw width makes the state scale-invariant: a standing adult has `a ≈ 0.4–0.6` regardless of their distance from the camera. This allows fixed, well-conditioned noise matrices Q and R, whereas raw width varies wildly with depth. Both SORT and ByteTrack use this parameterisation for the same reason.

**What was the double-increment bug?**
In v2, tracks moved from the active pool to the lost pool in Step A had their `missing` counter incremented. Then in Step B, all lost tracks (including the newly-added ones) were incremented again. So a track that missed exactly one frame got `missing=2` instead of `missing=1`, halving its effective occlusion lifetime and causing premature track termination and spurious ID switches.

**Why does the Market-1501 distractor fix matter?**
The official protocol keeps distractor images (pid==-1) in the gallery. They occupy rank positions without being counted as correct matches, making the task harder. Removing them from the gallery at load time (v2 bug) made the ranked list shorter and cleaner, artificially inflating Rank-1 by removing noise that would otherwise push true positives down the list. Fixed numbers are directly comparable to published results.
#   R E I D _ C V  
 