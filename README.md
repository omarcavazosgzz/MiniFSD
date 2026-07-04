# MiniFSD

MiniFSD is an end-to-end autonomous driving system built using CARLA and PyTorch.

## Features

- RGB Camera
- Imitation Learning
- Real-time Inference
- Autonomous Driving
- Route Planning
- Performance Evaluation

## Current Progress

- [x] Project initialized
- [x] CARLA environment
- [x] Data collection
- [x] Dataset generation
- [x] Model training
- [x] Autonomous driving (MEH)

## Dataset Collection

MiniFSD includes a keyboard-based data collection pipeline for imitation learning in CARLA.

The collector records:

- Front RGB camera images
- Steering
- Throttle
- Brake
- Vehicle speed

Each recording session is saved locally under:

```text
data/raw/<session_name>/
├── images/
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
└── driving_log.csv
```

Datasets are intentionally ignored by Git and should not be committed to the repository.

### Running CARLA Offscreen

To run CARLA without opening the Unreal window:

```bash
./scripts/dev/run_carla_offscreen.sh
```

This keeps rendering active for camera sensors while improving performance.

### Collecting Keyboard Driving Data

In another terminal:

```bash
cd ~/Projects/MiniFSD
source .venv/bin/activate

python scripts/collect_keyboard_data.py
```

Controls:

```text
W = throttle
A/D = steer
S = brake
R = pause/resume recording
Q = quit
```

Optional custom session name:

```bash
python scripts/collect_keyboard_data.py --session-name manual_test_001
```

### Analyzing a Dataset

Analyze the latest recorded session:

```bash
python scripts/analyze_dataset.py --latest
```

Analyze a specific session:

```bash
python scripts/analyze_dataset.py --session data/raw/manual_20260702_122903
```

A useful early dataset should include:

- Straight driving
- Left turns
- Right turns
- Some braking
- Moving frames with varied steering
- No missing images

Current first usable dataset:

```text
Session: manual_20260702_122903
Frames: 4190
Mean speed: 21.24 km/h
Steering balance: acceptable
Missing images: 0
```

### Milestone 2 Status

Milestone 2 completed the first usable data collection pipeline.

Implemented:

- DataRecorder class
- RGB camera frame saving
- Steering, throttle, brake, and speed logging
- Keyboard driving controller
- Automatic timestamped session folders
- Record/pause toggle
- Dataset summary and analysis script
- Offscreen CARLA launcher
- First usable driving dataset with 4190 frames

## Base Model Demo

This demo shows the first steering-only imitation learning model running in CARLA. The GIF below is a one-minute excerpt from the trial.

The [full, unedited 21-minute run](https://tecmx-my.sharepoint.com/:f:/g/personal/a00840617_tec_mx/IgCuw4TNVJvLT6pp_22xTbNpAX549yToU2zfISFpgKx8wCQ?e=aOMjoo) shows the base model completing the trial without crashing. This is still an early and incomplete baseline: it follows the map's outermost loop but does not enter the city, navigate intersections, or demonstrate dense inner-city driving.

This version is kept as the **Base Model — Outside Loop Demo** to document the first working end-to-end driving checkpoint.

### Demo

![Base Model Outside Loop](assets/gifs/base_model_outside_loop_small.gif)


## Training Logs

MiniFSD supports TensorBoard logging during training.

Example:

```bash
python scripts/train.py \
  --session data/raw/manual_20260702_122903 \
  --session data/raw/manual_recovery_001 \
  --epochs 15 \
  --batch-size 64 \
  --checkpoint-name steering_cnn_recovery_best.pt \
  --run-name baseline_recovery_v1
