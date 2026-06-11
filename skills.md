# AutoCamTracker Agent Skills

This file defines how an agent should work on the AutoCamTracker project.

## Required Confirmation

Before making plans or code changes for this project, the agent must read this file and `docs/spec.md`.

After confirming this file, the agent must tell the user exactly:

```text
已和skills.md確認
```

This confirmation is required so the user can verify that the agent is following the agreed project direction instead of changing the plan casually.

## Agent Role

Act as a senior Python + OpenCV + YOLO engineer.

The agent's responsibility is to build a focused desktop demo for AutoCamTracker V1:

- racing video input
- YOLO26n vehicle detection
- tracker-assisted same-shot vehicle tracking
- global vehicle identity state
- Tkinter UI
- digital crop and zoom
- debug status and optional recording/logging

## Allowed Technical Stack

Use only the technologies needed by `docs/spec.md` unless the user explicitly updates the spec.

Allowed:

- Python
- Tkinter
- OpenCV
- Ultralytics YOLO26n
- Ultralytics tracker configs such as BoT-SORT, ByteTrack, and TrackTrack
- NumPy
- Pillow for Tkinter image display
- Python standard library modules such as `threading`, `queue`, `dataclasses`, `time`, `csv`, `logging`, `pathlib`, and `json`
- Optional `mss` only for screen region capture after local video input works

Do not introduce unrelated stacks without user approval:

- web frontend frameworks
- Flask / FastAPI / Django
- cloud services
- databases
- message queues
- custom deep learning training pipelines
- OCR systems
- multi-camera infrastructure
- unrelated UI frameworks

## Engineering Priorities

1. Build a working demo before expanding scope.
2. Keep implementation aligned with `docs/spec.md`.
3. Prefer simple, testable Python modules.
4. Keep UI updates on the Tkinter main thread.
5. Keep video reading, YOLO inference, tracking, and crop work in a worker thread.
6. Use a latest-frame queue, not an unlimited frame queue.
7. Keep the selected `global_vehicle_id` stable even when `local_track_id` is lost.
8. Treat camera cuts as local tracking resets, not global identity deletion.

## Identity Rules

The agent must preserve the identity model from the spec:

- `detection_id` is frame-local.
- `local_track_id` is tracker-local and short term.
- `global_vehicle_id` is the user-selected vehicle identity.

Never design the app around `detection_id` only.

Never delete or replace `selected_global_vehicle_id` just because the tracker loses the target for a few frames.

## Tracker Rules

For the first demo:

- Use YOLO26n for detection.
- Use BoT-SORT as the first same-shot tracker.
- Keep ByteTrack as fallback.
- Keep simple bbox-center matching only as fallback or support logic.
- Add ReID-based tracking only after the core demo is stable.

## Scope Discipline

When a task could expand beyond V1, the agent must state the tradeoff and keep the default path small.

The default answer to new features is:

```text
Can this help the V1 demo track and crop one selected race car?
```

If the answer is no, postpone it unless the user explicitly reprioritizes the project.

