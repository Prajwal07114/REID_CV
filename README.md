# 🚀 Person Re-Identification & Multi-Camera Tracking System (v3.0)

A production-grade, modular pipeline for **multi-camera person tracking and re-identification** using modern deep learning and MOT techniques.

This system detects people across multiple camera streams, tracks them locally, extracts identity embeddings, and matches identities globally across cameras in real time.

---

# ✨ Features

- 🔍 **Person Detection** using YOLOv8
- 🎯 **Multi-Object Tracking** using ByteTrack + Kalman Filter
- 🧠 **Person Re-Identification (ReID)** using OSNet
- 🔗 **Cross-Camera Identity Matching**
- 📊 **Market-1501 Evaluation Support**
- ⚡ Batched multi-camera inference
- 📈 Ablation benchmarking framework
- 🎥 Real-time visualization HUD
- 🧪 Embedding sanity checks
- 🧱 Clean modular architecture

---

# 🧠 Pipeline Overview

```text
YOLOv8 Detection
        ↓
ByteTrack + Kalman Tracking
        ↓
OSNet Embedding Extraction
        ↓
Cosine Similarity Matching
        ↓
Global Cross-Camera Identity Assignment
```

---

# 🏗️ System Architecture

```text
┌──────────────────────────────────────────────┐
│                 main.py                      │
│          (Pipeline Orchestrator)             │
└───────────────┬───────────────┬──────────────┘
                │               │
        ┌───────▼───────┐ ┌─────▼────────┐
        │   Detection   │ │   Tracking   │
        │    YOLOv8     │ │ ByteTrack +  │
        │               │ │ Kalman Filter│
        └───────┬───────┘ └─────┬────────┘
                │               │
                └──────┬────────┘
                       ▼
             ┌──────────────────┐
             │       ReID       │
             │      OSNet       │
             │  Cosine Matching │
             └────────┬─────────┘
                      ▼
          ┌────────────────────────┐
          │ Cross-Camera Identity  │
          │     Global Gallery     │
          └────────┬───────────────┘
                   ▼
        ┌─────────────────────────┐
        │ Visualization & Metrics │
        └─────────────────────────┘
```

---

# 📁 Repository Structure

```text
reid-multicam-v3/
│
├── main.py
├── requirements.txt
├── README.md
│
├── config/
│   └── settings.py
│
├── detection/
│   └── detector.py
│
├── tracking/
│   ├── kalman.py
│   ├── tracker.py
│   └── tracklet.py
│
├── reid/
│   ├── embedding_engine.py
│   ├── osnet_engine.py
│   ├── resnet_engine.py
│   ├── gallery.py
│   └── pipeline.py
│
├── evaluation/
│   ├── market1501.py
│   └── metrics.py
│
├── benchmarking/
│   └── ablation.py
│
├── visualization/
│   └── hud.py
│
├── utils/
│   ├── io.py
│   ├── geometry.py
│   ├── logging_utils.py
│   └── sanity_check.py
│
└── outputs/
```

---

# ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Detection | YOLOv8 |
| Tracking | ByteTrack |
| Motion Model | Kalman Filter |
| ReID Backbone | OSNet |
| Matching | Cosine Similarity |
| Evaluation | Market-1501 |
| Framework | PyTorch |

---

# 🔧 Major Improvements in v3

| Fix | Description | Impact |
|---|---|---|
| ✅ OSNet loading fix | Proper Market-1501 pretrained embeddings | Accurate cosine matching |
| ✅ Market-1501 distractor fix | Keeps `pid==-1` gallery images | Benchmark correctness |
| ✅ ByteTrack bug fix | Prevents double increment of lost tracks | Fewer ID switches |
| ✅ Kalman redesign | Uses `[x,y,a,h]` state space | More stable tracking |
| ✅ Batched detection | Single YOLO call for all cameras | Faster inference |
| ✅ Gallery optimization | `deque(maxlen)` buffer | O(1) operations |
| ✅ Immutable config | Thread-safe architecture | Easier testing |

---

# 📦 Installation

## Clone Repository

```bash
git clone https://github.com/Prajwal07114/REID_CV.git
cd REID_CV
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ⚠️ torchreid Installation

If this fails:

```bash
pip install torchreid
```

Install from source instead:

```bash
git clone https://github.com/KaiyangZhou/deep-person-reid
cd deep-person-reid
pip install -e .
```

---

# 🚀 Usage

## Standard Multi-Camera Tracking

```bash
python main.py --cams cam1.mp4 cam2.mp4 cam3.mp4
```

## Select ReID Model

```bash
python main.py \
  --cams cam1.mp4 cam2.mp4 \
  --model osnet \
  --threshold 0.55
```

## Market-1501 Evaluation

```bash
python main.py \
  --cams cam1.mp4 \
  --eval \
  --eval-root /data/Market-1501
```

## Ablation Benchmarking

```bash
python main.py \
  --cams cam1.mp4 cam2.mp4 \
  --ablation \
  --models resnet18 resnet50 osnet \
  --trackers iou bytetrack
```

## Embedding Sanity Check

```bash
python main.py --sanity-check --model osnet
```

---

# 📊 Benchmark Results

| Backbone | Tracker | FPS | Global IDs | ID Switches |
|---|---|---|---|---|
| ResNet18 | IoU | — | — | — |
| ResNet18 | ByteTrack | — | — | — |
| ResNet50 | IoU | — | — | — |
| ResNet50 | ByteTrack | — | — | — |
| OSNet | IoU | — | — | — |
| OSNet | ByteTrack | — | — | — |

> Run `--ablation` to generate benchmark results on your hardware.

---

# 🧪 Market-1501 Evaluation

## Dataset Setup

1. Download the Market-1501 dataset
2. Extract it
3. Ensure the structure contains:

```text
query/
bounding_box_test/
```

## Run Evaluation

```bash
python main.py \
  --cams cam1.mp4 \
  --eval \
  --eval-root /path/to/Market-1501 \
  --model osnet
```

---

# 📈 Expected Performance

| Metric | Score |
|---|---|
| mAP | ~74% |
| Rank-1 | ~94% |
| Rank-5 | ~98% |
| Rank-10 | ~99% |

Using:
- OSNet-x1_0
- Market-1501 pretrained weights
- torchreid embeddings

---

# 🎯 How the System Works

## Step 1 — Person Detection

YOLOv8 detects all visible persons in each camera frame.

## Step 2 — Local Tracking

ByteTrack assigns stable local IDs using:
- IoU association
- Kalman motion prediction

## Step 3 — Feature Extraction

OSNet generates a discriminative embedding vector for every person crop.

## Step 4 — Cross-Camera Matching

Cosine similarity compares embeddings across cameras to determine whether two detections belong to the same identity.

## Step 5 — Global Identity Assignment

A unified global ID is assigned across all camera streams.

---

# 🧠 Technical Design Decisions

## Why OSNet Instead of ResNet?

OSNet is specifically trained for person ReID using metric learning losses like triplet loss.

This creates an embedding space where:
- Same identities cluster together
- Different identities remain far apart

ImageNet-trained ResNet models optimize category classification instead of identity discrimination, making them significantly weaker for ReID tasks.

---

## Why Use `[x, y, a, h]` in Kalman Tracking?

Where:
- `x, y` → center position
- `a` → aspect ratio
- `h` → height

Using aspect ratio instead of raw width provides:
- Better numerical stability
- Scale invariance
- Improved tracking consistency

This formulation is also used in:
- SORT
- DeepSORT
- ByteTrack

---

## ByteTrack Double-Increment Bug (v2)

### Problem

Lost tracks were incremented twice per frame.

### Effect

- Shortened occlusion lifetime
- Premature track deletion
- Increased ID switches

### Fix

Tracks now increment exactly once per missed frame.

---

## Market-1501 Distractor Fix

The official protocol includes distractor images (`pid == -1`) inside the gallery.

Removing them artificially inflates Rank-1 accuracy because:
- Fewer false candidates exist
- True matches move higher in ranking

v3 follows the official evaluation protocol correctly.

---

# 📌 Future Improvements

- [ ] DeepSORT integration
- [ ] TensorRT acceleration
- [ ] Distributed multi-node camera support
- [ ] FAISS ANN search for large galleries
- [ ] Transformer-based ReID backbones
- [ ] Web dashboard visualization
- [ ] Docker deployment

---

# 👨‍💻 Author

### Prajwal Barsagade

Computer Vision • Deep Learning • Multi-Object Tracking • Person ReID

---

# ⭐ If You Like This Project

Give the repository a star on GitHub ⭐

```bash
git clone https://github.com/Prajwal07114/REID_CV.git
```
