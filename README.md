# JumpLab — Single-Camera Jump Squat, Push-Up & Game Footage Analyzer

Analyze **jump squats**, **push-up form**, and **game footage** (basketball, volleyball,
soccer) from one ordinary video — entirely in your browser. AI pose estimation
(MediaPipe Pose Landmarker) runs locally; no video is uploaded to any server.

## What it measures

### Jump squat / game footage mode
| Metric | How |
|---|---|
| Vertical jump height (cm) | flight-time method (`h = g·t²/8`), calibrated to your real height |
| Takeoff velocity, mean power (W, W/kg) | from flight time + your bodyweight **and external load** |
| Countermovement depth & squat angle | knee/hip angle at the bottom of the dip |
| Foot position | stance width vs shoulder width (toe-out noted as camera-limited) |
| Arm distribution | L vs R arm-swing travel, overhead follow-through, symmetry |
| Head angle & trunk lean | shoulder-hip line off vertical; ear-shoulder alignment at apex |
| Triple extension | hip & knee extension at takeoff |
| Leg symmetry | L vs R ankle rise through the jump |

### Push-up mode
Rep count, elbow depth at bottom (90° target), lockout at top, hip sag/pike
(plank rigidity), left/right arm symmetry, tempo — plus a full corrective checklist.

### Every output is put in perspective
* Tier verdict vs **sedentary adult → recreational → competitive → elite** benchmarks,
  tuned to your selected **athletic level, load, and sport** (NBA/pro volleyball ~80–90 cm,
  competitive ~60–70, gym-goer ~40–50, general population ~25–35).
* Loaded jumps are scored on **W/kg power**, not raw height.
* Feedback written for **average runners, weightlifters and gym-goers** too — including
  carryover notes (clean/snatch second pull, running stiffness/economy, cardiac-risk
  context for push-up capacity).

## How to run
Because it uses ES modules + fetches the AI model, serve it over `http` (not `file://`):

```bash
cd jump-squat-analyzer
python -m http.server 8000
# or:  npx serve .
```

Open http://localhost:8000 (Chrome or Edge recommended; internet needed once to
download the ~10 MB pose model — analysis itself runs on your device).

## Filming tips (accuracy depends on these)
1. **Side view**, camera tripod-height, perpendicular to the movement plane.
2. Whole body inside frame at all times; athlete large in frame.
3. Steady camera — for game footage pick clips without zoom/pan; broadcast angles reduce accuracy (the app flags uncertain reads).
4. 3–10 second clips work best; stand still ~1 s before the rep so the app can calibrate the floor and baseline.
5. Enter your real height, bodyweight and load — they drive all physics estimates.

## Honest limitations
2-D single-camera estimates are coaching-grade, not force-plate grade. Occlusion
(other players), motion blur, high frame-rate TV footage, and camera panning will
degrade numbers. Use trends across sessions, not any single absolute value.
