# JumpLab — Single-Camera Jump Squat, Push-Up & Game Footage Analyzer

Analyze **jump squats**, **push-up form**, and **game footage** (basketball, volleyball,
soccer) from one ordinary video — entirely in your browser. AI pose estimation
(MediaPipe Pose Landmarker) runs locally; no video is uploaded to any server.

## What it measures

Units in the UI are US customary (lb, ft/in, inches); all physics is computed in SI and converted for display.

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

## Frame rate — what "120 fps" really means here

The Sample rate selector goes up to 240 fps, but a video file only contains the frames
the camera recorded. The app reads your file's real frame rate (from the decoder's
presentation timestamps via `requestVideoFrameCallback`) and shows it next to the
selector; sampling then stops at that ceiling and duplicate frames are discarded, so
120 fps on a 30 fps clip costs time but adds no information. To get genuinely higher
temporal detail, record in your phone's **slow-motion (120 or 240 fps)** mode and upload
that file — the app will report `source: 120 fps` and analyze every frame.

Webcam capture asks for `frameRate: { ideal: 120, max: 240 }`, records at whatever the
camera actually delivers, and tells you that number. Most webcams cap at 30 fps.

**Auto-trim (✂, on by default):** for clips over 4 s, the app first scans the whole
file at 12 fps to find where the rep actually is — feet-off-floor runs for jumps, elbow
minima for push-ups, joint motion energy as a fallback — then analyzes only that window.
It keeps 1.2 s of standing before the rep and 0.6 s after, because the floor line, the
standing baseline hip height and the start of the countermovement are all calibrated from
those frames; trimming to the exact takeoff frame would break the calibration, so that
tail is not dead time. On a 10 s clip with one jump this typically cuts ~5 s of standing
and walking, which is also what makes 120 fps affordable. The window used is reported in
the results and in the export, and playback loops just that window. Untick the box to
analyze the full file.

Known limits of the trimmer: with several jump attempts in one clip it keeps the
**highest** flight only; if the rep starts within ~1 s of the first frame the pre-roll is
clipped and the baseline comes from fewer frames; a clip with no clear live/dead contrast
(walking around the whole time) is left untrimmed rather than guessed at.

**Clip speed selector:** a slow-motion file is already time-stretched (e.g. 240 fps footage
saved as a 120 fps file plays at ½ speed, so its timestamps run half as fast as real time).
Pick ½ or ¼ speed in the UI and flight time / tempo are converted back to real seconds —
otherwise slow-mo clips report 4× the true jump height. Default is real time (1×).

Because flight time is what drives jump height, takeoff and landing are *interpolated
between* frames rather than snapped to them, so timing precision is better than
1/frame-rate even on 30 fps clips (a 300 ms flight is only ~9 frames at 30 fps).

## Where the numbers come from

1. **MediaPipe Pose Landmarker** (`pose_landmarker_full`, ~10 MB) runs locally in the
   browser (WASM + WebGL) and returns 33 2-D landmarks per analyzed frame with a
   visibility confidence. No video leaves your machine — the only network traffic is the
   one-time model + WASM download from a public CDN.
2. **Calibration:** pixel distances are converted to metres using *your* height (ankle →
   shoulder → nose pixel span divided by the height you entered).
3. **Geometry:** joint angles from landmark triples (dot product), trunk lean off
   vertical, hip deviation from the shoulder-ankle line for plank quality.
4. **Physics:** flight time → `h = g·t²/8`, `v₀ = g·t/2`, mean power `P = m·g·v₀` using
   bodyweight + entered load. Push-up reps are elbow-angle cycles (<110° bottom, >150° top).
5. **Scores / tiers:** hand-written coaching thresholds (90–110° knee at the bottom,
   170–180° plank, 30–45° trunk lean) weighted into the 0–100 score, plus published
   population/combine ranges hardcoded in `SPORT_BENCH`. Reference ranges, not a validated
   normative database.

The Technique Guide tab spells all of this out, including the limitations.

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
