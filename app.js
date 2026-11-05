/* =========================================================
   JumpLab — single-camera biomechanics analyzer
   Pose estimation: MediaPipe Pose Landmarker (runs locally in browser)
   ========================================================= */
import { PoseLandmarker, FilesetResolver } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const CDN_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task";

/* ---------- landmark indices ---------- */
const LM = {
  NOSE: 0, L_EAR: 7, R_EAR: 8,
  L_SHO: 11, R_SHO: 12, L_ELB: 13, R_ELB: 14, L_WRI: 15, R_WRI: 16,
  L_HIP: 23, R_HIP: 24, L_KNE: 25, R_KNE: 26, L_ANK: 27, R_ANK: 28,
  L_HEE: 29, R_HEE: 30, L_FOO: 31, R_FOO: 32
};

/* ---------- units: the UI is US customary, all math stays SI ---------- */
const LB_PER_KG = 2.2046226218, CM_PER_IN = 2.54;
const lbToKg = lb => lb / LB_PER_KG;
const kgToLb = kg => kg * LB_PER_KG;
const cmToIn = cm => cm / CM_PER_IN;
/* display helpers — imperial first, metric second */
const showIn = (cm, d = 1) => `${cmToIn(cm).toFixed(d)} in`;
const showCm = (cm, d = 1) => `${cm.toFixed(d)} cm`;
const showLb = kg => `${kgToLb(kg).toFixed(0)} lb`;
const showFtIn = cm => {
  const r = Math.round(cmToIn(cm));
  return `${Math.floor(r / 12)}'${r % 12}"`;
};
const heightCmFromInputs = () =>
  (+($("heightFt")?.value || 5) * 12 + +($("heightIn")?.value || 8)) * CM_PER_IN;

function initHeightSelects() {
  const ft = $("heightFt"), inch = $("heightIn");
  for (let f = 3; f <= 7; f++) ft.add(new Option(f, f));
  for (let i = 0; i < 12; i++) inch.add(new Option(i, i));
  ft.value = 5; inch.value = 10;                       // 5'10" ≈ 178 cm
  ft.onchange = inch.onchange = updateHeightEcho;
  updateHeightEcho();
}
function updateHeightEcho() {
  $("heightCmEcho").textContent = `(${Math.round(heightCmFromInputs())} cm)`;
}

/* ---------- DOM ---------- */
const $ = id => document.getElementById(id);
const srcVideo = $("srcVideo"), overlay = $("overlay"), octx = overlay.getContext("2d");
const statusMsg = $("statusMsg");

let landmarker = null;
let samples = [];          // [{t, pts, world}]  pts in pixel coords
let lastReport = null;     // for export
let activeTab = "jump";
let mediaRecorder = null, camStream = null;

/* =========================================================
   TABS
========================================================= */
document.querySelectorAll(".tab").forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    activeTab = btn.dataset.tab;
    $("guidePanel").classList.toggle("hidden", activeTab !== "guide");
    document.querySelector(".settings").classList.toggle("hidden", activeTab === "guide");
    document.querySelector(".capture").classList.toggle("hidden", activeTab === "guide");
    if (activeTab === "sport") $("captureTitle").textContent = "Game footage (basketball / volleyball / soccer …)";
    else if (activeTab === "pushup") $("captureTitle").textContent = "Push-up video (side view, whole body)";
    else $("captureTitle").textContent = "Jump squat video";
    $("results").classList.toggle("hidden", !lastReport || activeTab === "guide");
  };
});

/* =========================================================
   VIDEO INPUT
========================================================= */
$("fileInput").onchange = async e => {
  const f = e.target.files[0];
  if (!f) return;
  srcVideo.src = URL.createObjectURL(f);
  srcVideo.classList.remove("hidden");
  $("analyzeBtn").disabled = false;
  $("fpsInfo").textContent = "reading frame rate…";
  detectedFps = null; samplesInfo = null;
  status("Video loaded: " + f.name + ". Press Analyze.");
  const fps = await probeSourceFps();
  if (!fps) { $("fpsInfo").textContent = "frame rate unknown — will auto-cap while sampling"; return; }
  detectedFps = fps;
  $("fpsInfo").textContent = `source: ${fps} fps · ${srcVideo.videoWidth}×${srcVideo.videoHeight}`;
  // offer the file's own rate so "every frame" is one click away
  const sel = $("fpsSel");
  [...sel.options].filter(o => o.dataset.dynamic).forEach(o => o.remove());
  if (![...sel.options].some(o => +o.value === fps)) {
    const opt = new Option(`${fps} fps (every frame in this file)`, fps);
    opt.dataset.dynamic = "1";
    sel.add(opt);
  }
  sel.value = String(fps);
  status(`Video loaded: ${f.name} — ${fps} fps, ${srcVideo.videoWidth}×${srcVideo.videoHeight}. Press Analyze.`);
};

$("camBtn").onclick = async () => {
  let camFps = 30;
  try {
    camStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", frameRate: { ideal: 120, max: 240 } }, audio: false
    });
    camFps = Math.round(camStream.getVideoTracks()[0].getSettings().frameRate || 30);
  } catch (err) {
    status("Camera blocked: " + err.message, true); return;
  }
  const cam = $("camVideo");
  cam.srcObject = camStream; cam.play();
  $("camWrap").classList.remove("hidden");
  $("recBtn").classList.remove("hidden");
  $("camBtn").classList.add("hidden");
  $("camFpsInfo").classList.remove("hidden");
  $("camFpsInfo").textContent = `camera is live at ${camFps} fps — recording at that rate. ` +
    (camFps < 60 ? "Most webcams top out at 30 fps; for higher detail record slow-motion on a phone and upload the file." : "");
  cam.dataset.fps = camFps;
};

$("recBtn").onclick = () => {
  const cam = $("camVideo");
  const fps = +cam.dataset.fps || 30;
  const chunks = [];
  const rec = new MediaRecorder(cam.captureStream(fps));
  rec.ondataavailable = e => chunks.push(e.data);
  rec.onstop = () => {
    const blob = new Blob(chunks, { type: "video/webm" });
    srcVideo.src = URL.createObjectURL(blob);
    srcVideo.classList.remove("hidden");
    camStream?.getTracks().forEach(t => t.stop());
    $("camWrap").classList.add("hidden");
    $("recBtn").classList.add("hidden");
    $("camBtn").classList.remove("hidden");
    $("analyzeBtn").disabled = false;
    detectedFps = null; samplesInfo = null; $("fpsInfo").textContent = "";
    status(`Recording ready (${fps} fps webm). Press Analyze. Keep clips short (3–10 s is ideal).`);
  };
  rec.start();
  status("● Recording… perform the rep(s), then press Stop.");
  $("recBtn").textContent = "⏹ Stop recording";
  const stopOnce = () => { if (rec.state === "recording") rec.stop(); };
  $("recBtn").onclick = stopOnce;
};

/* =========================================================
   MODEL
========================================================= */
async function ensureModel() {
  if (landmarker) return;
  setProgress(0, "Loading AI pose model (~10 MB, one-time)…");
  const vision = await FilesetResolver.forVisionTasks(CDN_WASM);
  landmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO", numPoses: 1
  });
}

/* =========================================================
   FRAME SAMPLING
   A video file only holds the frames the camera actually recorded. We read the
   file's real frame rate first (requestVideoFrameCallback gives us the decoder's
   presentation timestamps) and cap sampling at it: asking for 120 fps from a
   30 fps clip would just run the model on the same decoded frame four times —
   slower, with no extra information.
========================================================= */
const waitEvent = (el, ev) => new Promise(r => el.addEventListener(ev, () => r(), { once: true }));
const nextFrame = v => v.requestVideoFrameCallback
  ? new Promise(r => v.requestVideoFrameCallback(r))    // resolves with (now, metadata)
  : new Promise(r => requestAnimationFrame(r));

const MAX_SAMPLES = 2600;   // cap on pose detections per clip (time budget)
const STD_RATES = [23.98, 24, 25, 30, 48, 50, 60, 90, 100, 120, 150, 200, 240];
const roundFps = f => STD_RATES.find(x => Math.abs(f - x) < x * 0.06) ?? Math.round(f);

let detectedFps = null;     // real frame rate of the loaded file
let samplesInfo = null;     // what we actually ended up sampling

/* Learn the file's frame rate by counting real decoded frames over ~0.7 s. */
async function probeSourceFps() {
  const v = srcVideo;
  if (!("requestVideoFrameCallback" in HTMLVideoElement.prototype)) return null;
  await new Promise(r => v.readyState >= 2 ? r() : waitEvent(v, "canplay").then(r));
  const t0 = v.currentTime;
  return new Promise(resolve => {
    const times = [];
    const wall0 = performance.now();
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      v.pause();
      v.currentTime = t0;
      const span = times.length > 1 ? times[times.length - 1] - times[0] : 0;
      resolve(times.length > 1 && span > 0.04 ? roundFps((times.length - 1) / span) : null);
    };
    const tick = (now, meta) => {
      times.push(meta.mediaTime);
      if (performance.now() - wall0 > 700 || v.ended) return finish();
      v.requestVideoFrameCallback(tick);
    };
    v.requestVideoFrameCallback(tick);
    v.play().catch(finish);
    setTimeout(finish, 3000);            // never leave the UI waiting
  });
}

/* Seek-sample the video between [from, to]. Shared by the coarse scan (which
   finds where the rep actually is) and the fine pass (which analyzes it). */
async function collectSamples({ fps, from = 0, to, label = "Tracking pose" }) {
  const v = srcVideo;
  const out = [];
  const step = 1 / fps;
  const span = Math.max(0.01, to - from);
  const wallStart = performance.now();
  let ts = performance.now(), lastT = -1;
  for (let t = from; t <= to; t += step) {
    const target = Math.min(t, v.duration - 0.001);
    v.currentTime = target;
    await waitEvent(v, "seeked");
    const meta = await nextFrame(v);
    const mediaTime = meta && Number.isFinite(meta.mediaTime) ? meta.mediaTime : null;
    const stamp = mediaTime !== null && mediaTime >= lastT ? mediaTime : target;
    if (lastT >= 0 && stamp - lastT < step * 0.45) continue;   // decoder served the same frame twice
    lastT = stamp;
    const res = landmarker.detectForVideo(v, ++ts);
    if (res.landmarks?.length) {
      const lm = res.landmarks[0];
      if ((lm[LM.L_HIP].visibility ?? 1) > 0.3 || (lm[LM.R_HIP].visibility ?? 1) > 0.3) {
        out.push({ t: stamp, pts: toPixels(lm, v), world: res.worldLandmarks?.[0] });
      }
    }
    const done = performance.now() - wallStart;
    const frac = span > 0 ? (target - from) / span : 1;
    const eta = frac > 0.05 ? Math.max(0, Math.round(done / frac * (1 - frac) / 1000)) : null;
    setProgress(frac, `${label} ${Math.round(100 * frac)}% · ${out.length} poses${eta !== null ? ` · ~${eta}s left` : ""}`);
  }
  return out;
}

/* Where is the rep? Scan the whole clip at a low rate, then keep the movement
   plus a short tail. That tail is not dead time: the floor line, the standing
   baseline hip height and the start of the countermovement all come from it. */
function findWindow(sims, mode, dur) {
  const n = sims.length;
  if (n < 8) return null;
  const times = sims.map(s => s.t);
  /* scale = ankle-mid to shoulder-mid ≈ 0.8 of standing height; every threshold
     below is a fraction of that, so it behaves the same for a 5 ft and a 6'8" athlete */
  const bodyPx = med(sims.map(s => dist(
    mid(s.pts[LM.L_ANK], s.pts[LM.R_ANK]), mid(s.pts[LM.L_SHO], s.pts[LM.R_SHO])) + 1)) || 100;

  let tA = null, tB = null, how = "";

  if (mode === "pushup") {
    const el = smooth(sims.map(s => (ang(s.pts[LM.L_SHO], s.pts[LM.L_ELB], s.pts[LM.L_WRI]) +
                                     ang(s.pts[LM.R_SHO], s.pts[LM.R_ELB], s.pts[LM.R_WRI])) / 2), 3);
    const lows = [];
    for (let i = 1; i < n - 1; i++) if (el[i] < 115 && el[i] <= el[i - 1] && el[i] <= el[i + 1]) lows.push(i);
    if (lows.length) { tA = times[lows[0]]; tB = times[lows.at(-1)]; how = `${lows.length} rep${lows.length > 1 ? "s" : ""} found`; }
  } else {
    // feet-off-floor runs: the biggest one is the jump
    const loY = sims.map(s => Math.max(s.pts[LM.L_ANK].y, s.pts[LM.R_ANK].y));
    const up = loY.map(y => (med(loY) - y) / bodyPx);           // body-lengths above the floor
    const runs = [];
    for (let i = 0; i < n; i++) {
      if (up[i] > 0.06) {                                        // ≈ 8 cm off the floor
        if (runs.length && i - runs.at(-1).b <= 2) runs.at(-1).b = i;
        else runs.push({ a: i, b: i });
      }
    }
    const best = runs.length ? runs.map(r => ({ ...r, peak: Math.max(...up.slice(r.a, r.b + 1)) }))
                                    .sort((x, y) => y.peak - x.peak)[0] : null;
    if (best && best.peak > 0.12) { tA = times[best.a]; tB = times[best.b]; how = "flight detected"; }   // ≈ 17 cm: a real jump, not jitter
  }

  if (tA === null) {
    // generic fallback: motion energy of the big joints
    const e = [0];
    for (let i = 1; i < n; i++) {
      const dt = Math.max(1e-3, times[i] - times[i - 1]);
      let m = 0;
      for (const k of [LM.L_HIP, LM.R_HIP, LM.L_KNE, LM.R_KNE, LM.L_SHO, LM.R_SHO, LM.L_WRI, LM.R_WRI])
        m += dist(sims[i].pts[k], sims[i - 1].pts[k]);
      e.push(m / 8 / bodyPx / dt);                              // body-lengths per second
    }
    const sm = smooth(e, 5);
    const peak = Math.max(...sm);
    /* Level is set from the clip's TYPICAL activity, not its max: one bad frame
       (a limb swap, a cut, a spectator crossing by) would otherwise set the bar
       so high that the real movement falls under it. */
    if (peak < 0.25) return null;                               // nothing moving: don't trim
    const lvl = Math.max(0.25, med(sm) * 1.25);
    const hot = sm.map((v, i) => v > lvl ? i : -1).filter(i => i >= 0);
    if (!hot.length) return null;
    /* Only trust the window if it is clearly more active than the rest of the
       clip — uniform landmark jitter has no live/dead contrast and must not trim. */
    const a = hot[0], b = hot.at(-1);
    const meanIn = avg(sm.slice(a, b + 1));
    const outside = [...sm.slice(0, a), ...sm.slice(b + 1)];
    if (outside.length < 3 || meanIn < avg(outside) * 2 + 0.15) return null;
    tA = times[a]; tB = times[b]; how = "motion detected";
  }

  const pre = mode === "pushup" ? 0.6 : 1.2, post = 0.6;
  const from = Math.max(0, tA - pre);
  const to = Math.min(dur, Math.max(tB + post, from + 1.5));
  if (to - from > dur * 0.85) return null;                      // not enough dead time to bother
  return { from, to, how };
}

async function sampleVideo(fpsReq, mode, doTrim) {
  const v = srcVideo;
  await new Promise(r => v.readyState >= 1 ? r() : waitEvent(v, "loadedmetadata"));
  const dur = Math.min(v.duration, 30);
  if (!detectedFps) detectedFps = await probeSourceFps();
  const srcFps = detectedFps;
  const capped = !!srcFps && fpsReq > srcFps * 1.05;

  let from = 0, to = dur, trim = null;
  if (doTrim && dur > 4) {
    const scanFps = Math.min(12, srcFps || 12);
    const coarse = await collectSamples({ fps: scanFps, from: 0, to: dur, label: "Scanning for the rep…" });
    const w = coarse.length > 8 ? findWindow(coarse, mode, dur) : null;
    if (w) {
      from = w.from; to = w.to;
      trim = { from, to, trimmed: dur - (to - from), how: w.how, scanFps };
    }
  }

  let fps = srcFps ? Math.min(fpsReq, srcFps * 1.05) : fpsReq;
  if ((to - from) * fps > MAX_SAMPLES) fps = Math.max(10, MAX_SAMPLES / (to - from));

  const out = await collectSamples({ fps, from, to, label: "Tracking pose…" });
  const realFps = out.length > 1 ? (out.length - 1) / (out.at(-1).t - out[0].t) : 0;
  samplesInfo = {
    requested: fpsReq, source: srcFps, capped,
    effective: realFps ? Math.round(realFps) : 0, frames: out.length, trim
  };
  return out;
}

const trimMetrics = () => samplesInfo?.trim ? [[
  "Clip window", `${samplesInfo.trim.from.toFixed(2)}–${samplesInfo.trim.to.toFixed(2)} s`,
  `cut ${samplesInfo.trim.trimmed.toFixed(1)} s of dead time · ${samplesInfo.trim.how} · scanned at ${samplesInfo.trim.scanFps} fps`
]] : [];

function toPixels(lm, v) {
  return lm.map(p => ({ x: p.x * v.videoWidth, y: p.y * v.videoHeight, v: p.visibility ?? 1 }));
}

/* =========================================================
   GEOMETRY HELPERS
========================================================= */
const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, v: Math.min(a.v, b.v) });
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
function ang(a, b, c) { // angle at vertex b, degrees
  const v1x = a.x - b.x, v1y = a.y - b.y, v2x = c.x - b.x, v2y = c.y - b.y;
  const d = (v1x * v2x + v1y * v2y) / (Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y) + 1e-9);
  return Math.acos(Math.min(1, Math.max(-1, d))) * 180 / Math.PI;
}
const avg = a => a.reduce((s, x) => s + x, 0) / (a.length || 1);
const med = a => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1] ?? 0; };
function smooth(a, w = 5) {
  const h = w >> 1, out = [];
  for (let i = 0; i < a.length; i++) {
    let s = 0, n = 0;
    for (let j = i - h; j <= i + h; j++) if (a[j] !== undefined) { s += a[j]; n++; }
    out.push(s / n);
  }
  return out;
}
function perFrame(samples, fn) { return samples.map(s => fn(s.pts)); }

/* body scale: pixels-per-meter from user's real height */
function bodyScale(samples, heightCm) {
  const px = samples.map(s => {
    const ankleMid = mid(s.pts[LM.L_ANK], s.pts[LM.R_ANK]);
    const shMid = mid(s.pts[LM.L_SHO], s.pts[LM.R_SHO]);
    return dist(s.pts[LM.NOSE], shMid) + dist(shMid, ankleMid);
  });
  const bodyPx = med(px.filter(p => p > 10)) || 1;
  return bodyPx / (heightCm / 100); // px per meter
}

/* =========================================================
   JUMP SQUAT / GAME FOOTAGE ANALYSIS
========================================================= */
const SPORT_BENCH = {
  general:      { elite: 75, comp: 62 },
  basketball:   { elite: 85, comp: 70 },
  volleyball:   { elite: 85, comp: 68 },
  soccer:       { elite: 72, comp: 58 },
  running:      { elite: 65, comp: 52 },
  weightlifting:{ elite: 80, comp: 65 }
};

function analyzeJump(samples, cfg) {
  const n = samples.length;
  if (n < 15) throw "Not enough pose data — use a clip where the full body is visible for at least ~1 s.";
  const ppm = bodyScale(samples, cfg.heightCm);
  const g = 9.81;

  /* Floor line and standing baseline come from the QUIET frames (feet on the
     floor), not from a fixed opening slice — the window we analyze is now
     auto-trimmed to the rep, so the first 25% isn't guaranteed to be standing. */
  const loYs = samples.map(s => Math.max(s.pts[LM.L_ANK].y, s.pts[LM.R_ANK].y));
  const quiet = loYs.map((y, i) => Math.abs(y - med(loYs)) < 0.06 * ppm ? i : -1).filter(i => i >= 0);
  const stand = quiet.length >= 5 ? quiet
    : Array.from({ length: Math.max(5, Math.floor(n * 0.25)) }, (_, i) => i);
  const floorY = med(stand.map(i => loYs[i]));
  const hipHm = samples.map(s => (floorY - mid(s.pts[LM.L_HIP], s.pts[LM.R_HIP]).y) / ppm);
  const hip = smooth(hipHm, 5);
  const base = med(stand.map(i => hip[i]));

  const footUp = samples.map(s => {
    const lo = Math.max(s.pts[LM.L_ANK].y, s.pts[LM.R_ANK].y);
    return (floorY - lo) / ppm;
  });

  // apex of hip
  let apex = 0; hip.forEach((h, i) => { if (h > hip[apex]) apex = i; });
  if (apex < 3 || apex > n - 4) throw "Couldn't find a clear jump — start the clip with the athlete standing, then perform the jump squat.";

  // flight window around apex (both feet off floor). Takeoff and landing are
  // interpolated BETWEEN frames, so flight time is not limited to whole frames.
  const thr = Math.max(0.05, 0.045 * (cfg.heightCm / 100));
  let i0 = apex, i1 = apex;
  while (i0 > 0 && footUp[i0] > thr) i0--;
  while (i1 < n - 1 && footUp[i1] > thr) i1++;
  const crossing = (a, b, level) => {          // time the foot trace passes `level`
    const d = footUp[b] - footUp[a];
    if (!Number.isFinite(d) || Math.abs(d) < 1e-6) return samples[b].t;
    const f = Math.max(0, Math.min(1, (level - footUp[a]) / d));
    return samples[a].t + f * (samples[b].t - samples[a].t);
  };
  const tTakeoff = crossing(i0, i0 + 1, thr);
  const tLand = crossing(i1 - 1 < i0 ? i0 : i1 - 1, i1, thr);
  /* slow-mo files are time-stretched: kT = real seconds per file second */
  const kT = cfg.timeScale || 1;
  const tFlight = Math.max(0, tLand - tTakeoff) * kT;
  let jumpH, method;
  if (tFlight >= 0.12 && tFlight <= 1.2) { jumpH = g * tFlight * tFlight / 8; method = "flight time (sub-frame interpolated)"; }
  else { // fallback: hip rise from last pre-apex local min before explosive drive
    let takeoff = apex; while (takeoff > 0 && hip[takeoff - 1] < hip[takeoff]) takeoff--;
    jumpH = hip[apex] - hip[Math.max(0, takeoff - 1)]; method = "hip displacement (approx.)";
  }
  const jumpCm = jumpH * 100;

  // crouch (deepest hip before apex)
  let crouch = 0; hip.forEach((h, i) => { if (i <= apex && h < hip[crouch]) crouch = i; });
  const depthCm = (base - hip[crouch]) * 100;
  const parallaxOK = depthCm > 8;

  const at = i => samples[Math.max(0, Math.min(n - 1, i))].pts;
  const cP = at(crouch), tP = at(Math.max(0, i0 - 1)), aP = at(apex);
  const kneeL = ang(cP[LM.L_ANK], cP[LM.L_KNE], cP[LM.L_HIP]);
  const kneeR = ang(cP[LM.R_ANK], cP[LM.R_KNE], cP[LM.R_HIP]);
  const knee = avg([kneeL, kneeR]);
  const hipAng = ang(cP[LM.L_SHO], cP[LM.L_HIP], cP[LM.L_KNE]);
  const stanceCm = Math.abs(cP[LM.L_ANK].x - cP[LM.R_ANK].x) / ppm * 100;
  const shW = dist(cP[LM.L_SHO], cP[LM.R_SHO]) / ppm * 100;

  // trunk lean & head at takeoff: shoulder-mid vs hip-mid, off vertical
  const trunkLean = i => {
    const p = at(i), sh = mid(p[LM.L_SHO], p[LM.R_SHO]), hp = mid(p[LM.L_HIP], p[LM.R_HIP]);
    return Math.atan2(sh.x - hp.x, hp.y - sh.y) * 180 / Math.PI;
  };
  const leanTakeoff = trunkLean(Math.max(0, i0 - 1)), leanApex = trunkLean(apex);
  // head tilt: ear-mid vs shoulder-mid off vertical (neutral head = small value)
  const headTilt = (() => {
    const p = aP, e = mid(p[LM.L_EAR], p[LM.R_EAR]), sh = mid(p[LM.L_SHO], p[LM.R_SHO]);
    if (e.v < 0.3) return null;
    return Math.abs(Math.atan2(e.x - sh.x, sh.y - e.y) * 180 / Math.PI);
  })();

  // extension at apex (shoulder-hip-knee and hip-knee-ankle)
  const extHip = ang(aP[LM.L_SHO], aP[LM.L_HIP], aP[LM.L_KNE]);
  const extKnee = ang(aP[LM.L_HIP], aP[LM.L_KNE], aP[LM.L_ANK]);

  // arm drive: wrist vertical travel from crouch→takeoff and symmetry
  const wr0 = at(Math.max(0, crouch - 3)), wr1 = at(Math.max(0, i0 - 1));
  const armL = (wr0[LM.L_WRI].y - wr1[LM.L_WRI].y) / ppm * 100;
  const armR = (wr0[LM.R_WRI].y - wr1[LM.R_WRI].y) / ppm * 100;
  const armSym = 100 - Math.min(100, Math.abs(armL - armR) / Math.max(1, Math.max(armL, armR)) * 100);
  const armOverHead = aP[LM.L_WRI].y < aP[LM.L_SHO].y && aP[LM.R_WRI].y < aP[LM.R_SHO].y;

  // leg symmetry across whole rep: difference in ankle rise
  const ankleRiseL = med(samples.slice(crouch, i1 + 1).map(s => (floorY - s.pts[LM.L_ANK].y) / ppm));
  const ankleRiseR = med(samples.slice(crouch, i1 + 1).map(s => (floorY - s.pts[LM.R_ANK].y) / ppm));
  const legSym = 100 - Math.min(100, Math.abs(ankleRiseL - ankleRiseR) / Math.max(0.05, Math.max(ankleRiseL, ankleRiseR)) * 100);

  // physics
  const mass = cfg.bodyweightKg + cfg.loadKg;
  const v0 = g * tFlight / 2 || Math.sqrt(2 * g * jumpH);
  const powerW = mass * g * v0;                  // mean propulsive power estimate
  const wattsPerKg = powerW / cfg.bodyweightKg;
  const rateRPE = jumpCm > 0 ? (cfg.loadKg > 0 ? "loaded" : "bodyweight") : "n/a";

  /* ---------- scoring ---------- */
  const bench = SPORT_BENCH[cfg.sport] || SPORT_BENCH.general;
  const levelExp = { beginner: 30, recreational: 42, advanced: 55, elite: 68 }[cfg.level];
  const loadPenalty = cfg.loadKg > 0 ? Math.min(15, cfg.loadKg / mass * 40) : 0; // loaded jumps are lower — normalize expectation
  const heightScore = clamp01(jumpCm / (levelExp + loadPenalty + 8)) * 100;

  const depthScore = depthCm < 5 ? 10 :
    knee > 130 ? 35 : knee > 110 ? 70 : knee >= 80 ? 100 : knee >= 60 ? 85 : 65;
  const extScore = clamp01(((extHip + extKnee) / 2 - 140) / 35) * 100;
  const trunkScore = Math.abs(leanTakeoff) < 8 ? 70 :
    Math.abs(leanTakeoff) <= 50 ? 100 : 60;
  const headScore = headTilt === null ? 80 : headTilt < 15 ? 100 : headTilt < 30 ? 80 : 55;
  const stanceScore = Math.abs(stanceCm - shW) < 8 ? 100 : Math.abs(stanceCm - shW) < 18 ? 75 : 50;

  const subs = {
    "Explosiveness / jump height": Math.round(heightScore),
    "Squat depth": Math.round(depthScore),
    "Full extension at takeoff": Math.round(extScore),
    "Trunk & head position": Math.round((trunkScore + headScore) / 2),
    "Arm drive & symmetry": Math.round((clamp01(Math.max(armL, armR) / 45) * 100 + armSym) / 2),
    "Leg symmetry": Math.round(legSym),
    "Stance width": Math.round(stanceScore)
  };
  const weights = { 0: .28, 1: .15, 2: .15, 3: .12, 4: .12, 5: .12, 6: .06 };
  let score = 0, wi = 0;
  for (const k in subs) { score += subs[k] * (weights[wi] ?? .1); wi++; }

  const rep = {
    mode: cfg.sport === "general" ? "Jump squat" : `Jump analysis (${cfg.sport})`,
    score: Math.round(score), subs, method,
    metrics: [
      ...trimMetrics(),
      ["Vertical jump height", showIn(jumpCm), `${showCm(jumpCm)} · ${method}`],
      ["Flight time", tFlight ? `${(tFlight * 1000).toFixed(0)} ms` : "not detected", samplesInfo ? `sampled at ${samplesInfo.effective} fps` : ""],
      ["Countermovement depth", showIn(depthCm, 1), `${showCm(depthCm)} · ${parallaxOK ? "clear dip" : "⚠ shallow/no crouch"}`],
      ["Knee angle at bottom", `${knee.toFixed(0)}°`, knee > 130 ? "shallow" : knee >= 80 ? "good" : "deep"],
      ["Trunk lean @ takeoff", `${leanTakeoff.toFixed(0)}°`, ""],
      ["Head tilt @ apex", headTilt === null ? "n/a" : `${headTilt.toFixed(0)}°`, ""],
      ["Extension @ apex (hip/knee)", `${extHip.toFixed(0)}° / ${extKnee.toFixed(0)}°`, ""],
      ["Stance width", showIn(stanceCm, 1), `shoulders ${showIn(shW, 1)}`],
      ["Arm swing travel (L/R)", `${showIn(armL, 1)} / ${showIn(armR, 1)}`, armOverHead ? "full follow-through" : "no overhead follow-through"],
      ["Est. takeoff velocity", `${(v0 * 2.23694).toFixed(1)} mph`, `${v0.toFixed(2)} m/s`],
      ["Est. mean power", `${powerW.toFixed(0)} W`, `${wattsPerKg.toFixed(1)} W/kg · ${showLb(cfg.bodyweightKg)} bodyweight`],
      ["Leg symmetry", `${legSym.toFixed(0)}%`, ""],
      ["Mass used for power", showLb(mass), `body ${showLb(cfg.bodyweightKg)} + load ${showLb(cfg.loadKg)}`],
      ["Load", cfg.loadKg > 0 ? showLb(cfg.loadKg) : showLb(0), rateRPE],
      ["Video frame rate", samplesInfo ? (samplesInfo.source ? `${samplesInfo.source} fps` : "unknown") : "n/a",
        samplesInfo ? `${samplesInfo.frames} poses analyzed` : ""],
      ["Clip time base", kT === 1 ? "real time" : `${kT.toFixed(2)}× (slow-mo corrected)`,
        kT === 1 ? "no correction applied" : "flight time converted to real seconds"]
    ]
  };

  rep.perspective = jumpPerspective(jumpCm, cfg, bench, levelExp);
  rep.feedback = jumpFeedback(subs, { knee, depthCm, leanTakeoff, headTilt, stanceCm, shW,
    armL, armR, armSym, armOverHead, extHip, extKnee, jumpCm, levelExp, cfg, legSym, cfg_sport: cfg.sport });
  return rep;
}

const clamp01 = x => Math.max(0, Math.min(1, x));

function jumpPerspective(cm, cfg, bench, levelExp) {
  const rows = [];
  const pct = Math.round(clamp01(cm / 95) * 100);
  let tier, tierTxt;
  if (cm >= bench.elite)      { tier = "elite";    tierTxt = `Elite ${cfg.sport} level — you're clearing the pro/college benchmark (${cmToIn(bench.elite).toFixed(0)}+ in / ${bench.elite}+ cm).`; }
  else if (cm >= bench.comp)  { tier = "comp";     tierTxt = `Competitive ${cfg.sport} level (benchmark ${cmToIn(bench.comp).toFixed(0)}–${cmToIn(bench.elite).toFixed(0)} in).`; }
  else if (cm >= levelExp)    { tier = "level";    tierTxt = `At/above what's expected for a ${cfg.level} athlete of your profile (~${cmToIn(levelExp).toFixed(0)} in / ${levelExp} cm).`; }
  else if (cm >= levelExp*0.7){ tier = "near";     tierTxt = `Below the ${cfg.level} benchmark (~${cmToIn(levelExp).toFixed(0)} in) — within reach with 6–8 weeks of plyometric work.`; }
  else                        { tier = "develop";  tierTxt = `Development stage — biggest wins come from depth quality and arm drive right now.`; }
  rows.push({ tier, txt: tierTxt });
  rows.push({ tier: "info", txt: `Rough population scale (inches, cm in parentheses): average sedentary adult ~10–14 in (25–35 cm), regular gym-goer ~16–20 in (40–50 cm), competitive athlete ~22–28 in (55–70 cm), elite ${cfg.sport} players ~${cmToIn(bench.elite).toFixed(0)}+ in (${bench.elite}+ cm). Your ${showIn(cm, 1)} (${showCm(cm)}) ≈ top ${100 - pct}% of a 0–95 cm / 0–37 in scale.` });
  if (cfg.loadKg > 0) rows.push({ tier: "info", txt: `You jumped with ${showLb(cfg.loadKg)} (${cfg.loadKg.toFixed(1)} kg) added — loaded jumps are naturally lower. Compare power (W/kg), not raw height, between loaded and bodyweight sessions.` });
  rows.push({ tier: "info", txt: `For the average person: each 2 in (5 cm) of vertical ≈ noticeably better running economy, easier stair/curve sprints, and far safer landings as you age. For lifters: jump-squat power carries directly into the second pull of cleans/snatches and lockout speed.` });
  return rows;
}

function jumpFeedback(s, m) {
  const F = [];
  const add = (level, area, tip) => F.push({ level, area, tip });
  const sp = { general: "", basketball: " In basketball this is your block/layoff jump.", volleyball: " In volleyball this is your block/spike jump.", soccer: " In soccer this is your header jump.", running: "", weightlifting: "" }[m.cfg_sport] || "";

  // depth
  if (m.knee > 130) add("bad", "Squat angle", `Your bottom position was shallow (knee ~${m.knee.toFixed(0)}°, only ${showIn(m.depthCm, 1)} of dip). A jump squat should match your competition squat depth — aim for knee angle 90–110°. Deeper = more muscle stretch = more free power. Practice: bodyweight jump squats to a box at parallel.` + sp);
  else if (m.knee > 110) add("ok", "Squat angle", `Decent dip (~${m.knee.toFixed(0)}° knee). You'd gain height by getting 10–15° deeper — drive hips between the knees, chest proud.`);
  else if (m.knee >= 60) add("good", "Squat angle", `Excellent countermovement depth (${m.knee.toFixed(0)}° knee, ${showIn(m.depthCm, 1)}). That full stretch-reflex is where effortless height lives.`);
  else add("warn", "Squat angle", `Very deep bottom (${m.knee.toFixed(0)}°). Fine for mobility work, but the ideal jump-squat bottom is 90–110° knee — any deeper and you leak energy out of the hole.`);

  // foot position / stance
  if (Math.abs(m.stanceCm - m.shW) < 8) add("good", "Foot position", `Stance width (${showIn(m.stanceCm, 1)}) matched your shoulder width — ideal. Keep toes turned out 5–15° and knees tracking over the 2nd–3rd toe.`);
  else if (m.stanceCm > m.shW + 8) add("ok", "Foot position", `Feet were wider than shoulders (${showIn(m.stanceCm, 1)} vs ${showIn(m.shW, 1)}). Narrow toward shoulder width: a wide stance shortens the push-off and reduces vertical force.`);
  else add("ok", "Foot position", `Stance narrower than shoulders (${showIn(m.stanceCm, 1)} vs ${showIn(m.shW, 1)}). Widen to shoulder width, toes slightly out — you'll feel more glute engagement and a stabler takeoff.`);
  add("info", "Foot position", `Single-camera note: toe-out angle isn't reliably visible from ${m.cfg.view || "this"} view; film 45° front-side if you want that checked.`);

  // arm drive
  const armMax = Math.max(m.armL, m.armR);
  const ARM_TRAVEL_MIN_CM = 20;   // ~8 in of wrist travel; thresholds stay in SI
  if (armMax < ARM_TRAVEL_MIN_CM) add("bad", "Arm distribution", `Minimal arm swing (only ${showIn(armMax, 1)} of travel). Arms contribute 10–15% of jump height: sweep them down-back as you dip, then explode them up past the ears at takeoff. Drill: pause at the bottom, shout 'up!' and throw arms first.`);
  else if (!m.armOverHead) add("ok", "Arm distribution", `You swung arms (${showIn(armMax, 1)}) but didn't follow through overhead. Release late and let the shoulders rise — wrists past ear height at takeoff buys extra height.`);
  else if (m.armSym < 80) add("warn", "Arm distribution", `Arms are uneven (L ${showIn(m.armL, 1)} vs R ${showIn(m.armR, 1)}, ${m.armSym.toFixed(0)}% symmetry). Asymmetry bleeds force sideways. Single-arm dumbbell jumps (light!) and banded arm-drive drills even this out.`);
  else add("good", "Arm distribution", `Full, symmetrical arm drive (${showIn(armMax, 1)} of travel, wrists past shoulders at takeoff) — textbook. Keep that timing: arms start moving ~0.1 s before the legs reverse direction.`);

  // trunk & head
  if (Math.abs(m.leanTakeoff) > 50) add("bad", "Head/trunk", `Trunk pitched ${m.leanTakeoff.toFixed(0)}° forward at takeoff — force went forward, not up. Keep chest over mid-foot; a 30–45° lean is powerful, 50°+ is a long jump, not a high one.`);
  else if (Math.abs(m.leanTakeoff) < 10) add("ok", "Head/trunk", `Very upright takeoff (${m.leanTakeoff.toFixed(0)}°). Fine, but a slight 30–40° shin angle/forward set lets the glutes work — don't fear the lean, control it.`);
  if (m.headTilt !== null && m.headTilt > 30) add("warn", "Head/trunk", `Head was cranked ${m.headTilt.toFixed(0)}° off neutral at the top. Where the head goes, the spine follows: eyes on a fixed point (rim/net/ball), chin level. This also protects the neck on contact/landing.` + sp);
  else add("good", "Head/trunk", `Head stayed in line with the spine — neutral neck keeps the whole posterior chain firing in sequence.`);

  // extension
  const extAvg = (m.extHip + m.extKnee) / 2;
  if (extAvg < 155) add("ok", "Extension", `Takeoff wasn't fully extended (hip/knee ${m.extHip.toFixed(0)}/${m.extKnee.toFixed(0)}°). Finish the triple extension — hips forward, knees locked, up on the toes at the instant you leave the ground. 'Tall at takeoff'.`);
  else add("good", "Extension", `Clean triple extension (hip/knee ${m.extHip.toFixed(0)}/${m.extKnee.toFixed(0)}°) — you used the whole body, not just the quads.`);

  // symmetry
  if (m.legSym < 80) add("warn", "Symmetry", `Legs pushed unevenly (${m.legSym.toFixed(0)}% symmetry). With single-camera video the near leg reads higher, so re-test from the other side before worrying — but if it repeats, add split squats and single-leg calf raises; takeoff asymmetry is a top predictor of landing injuries in ${m.cfg_sport === "general" ? "sport" : m.cfg_sport}.`);

  // height context by level
  if (m.jumpCm < m.levelExp * 0.7) add("info", "Programming", `To move from ${showIn(m.jumpCm, 1)} toward ${showIn(m.levelExp, 1)}+ as a ${m.cfg.level} athlete: 2×/week plyos (pogo hops 3×20, drop jumps 4×5), jump squats 4×5 at 0–30% bodyweight load, and heavy trap-bar or back squats 4×5 @ 80–85%. Expect +1–2.5 in (3–6 cm) in 8 weeks.`);
  else if (m.jumpCm >= m.levelExp) add("good", "Programming", `You're hitting ${m.cfg.level}-level height (${showIn(m.jumpCm, 1)}). To keep climbing, train contrast sets: heavy squat 3×3 @85% → rest 20 s → max jump 5×1, 2×/week. Track W/kg — it's the metric that keeps improving after height plateaus.`);

  if (m.cfg_sport === "weightlifting" || m.cfg.sport === "weightlifting") add("info", "Sport carryover", `For lifting: this same triple-extension pattern IS your clean/snatch second pull. Keep loads in the 30–60% 1RM jump-squat range for velocity, 0–30% for pure jump height.`);
  else if (m.cfg_sport === "running") add("info", "Sport carryover", `For running: your elastic spring is what these build. Keep volume low (3–5 sets × 5 reps), full recovery between sets — you want quality springs, not conditioning. Stiff-legged pogo hops transfer most directly to cadence and economy.`);
  return F;
}

/* =========================================================
   PUSH-UP ANALYSIS
========================================================= */
function analyzePushUp(samples, cfg) {
  const n = samples.length;
  if (n < 15) throw "Not enough data — record at least 3 full push-up reps.";
  const ppm = bodyScale(samples, cfg.heightCm);

  const elbowL = smooth(perFrame(samples, p => ang(p[LM.L_SHO], p[LM.L_ELB], p[LM.L_WRI])), 5);
  const elbowR = smooth(perFrame(samples, p => ang(p[LM.R_SHO], p[LM.R_ELB], p[LM.R_WRI])), 5);
  const elbow = elbowL.map((e, i) => (e + elbowR[i]) / 2);
  const bodyH = cfg.heightCm / 100;

  // rep cycles: top >150°, bottom <110°
  const bottoms = [], tops = [];
  let prev = elbow[0];
  for (let i = 2; i < n - 2; i++) {
    const isBot = elbow[i] < 110 && elbow[i] <= elbow[i - 1] && elbow[i] <= elbow[i + 1];
    const isTop = elbow[i] > 150 && elbow[i] >= elbow[i - 1] && elbow[i] >= elbow[i + 1];
    if (isBot && (bottoms.length === 0 || i - bottoms.at(-1) > 4)) bottoms.push(i);
    if (isTop && (tops.length === 0 || i - tops.at(-1) > 4)) tops.push(i);
  }
  const reps = Math.min(bottoms.length, Math.max(0, tops.length - (tops[0] < (bottoms[0] ?? 1e9) ? 1 : 0)));
  if (bottoms.length === 0) throw "No push-up reps detected — lower until elbows pass 90° and keep the whole body in frame.";

  // plank quality: hip deviation from shoulder→ankle line at bottoms
  const sags = bottoms.map(i => {
    const p = samples[i].pts, sh = mid(p[LM.L_SHO], p[LM.R_SHO]), an = mid(p[LM.L_ANK], p[LM.R_ANK]), hp = mid(p[LM.L_HIP], p[LM.R_HIP]);
    const L = dist(sh, an); if (L < 1) return 0;
    const t = Math.max(0, Math.min(1, ((hp.x - sh.x) * (an.x - sh.x) + (hp.y - sh.y) * (an.y - sh.y)) / (L * L)));
    const proj = { x: sh.x + t * (an.x - sh.x), y: sh.y + t * (an.y - sh.y) };
    return (dist(hp, proj) / ppm); // meters of deviation
  });
  const hipLine = med(sags); // <0.04 great
  const bodyAng = bottoms.map(i => {
    const p = samples[i].pts;
    return ang(mid(p[LM.L_SHO], p[LM.R_SHO]), mid(p[LM.L_HIP], p[LM.R_HIP]), mid(p[LM.L_ANK], p[LM.R_ANK]));
  });
  const minBodyAng = Math.min(...bodyAng);

  const depths = bottoms.map(i => elbow[i]);
  const depthMed = med(depths);
  const symmetry = med(bottoms.map(i => Math.abs(elbowL[i] - elbowR[i])));

  // tempo (slow-mo files are time-stretched — convert to real seconds)
  const kT = cfg.timeScale || 1;
  const repTimes = [];
  for (let k = 1; k < bottoms.length; k++) repTimes.push((samples[bottoms[k]].t - samples[bottoms[k - 1]].t) * kT);
  const tempo = repTimes.length ? med(repTimes) : null;

  // lockout at top
  const topMed = tops.length ? med(tops.map(i => elbow[i])) : 180;

  // hand vs shoulder (front-view only meaningful; report offset anyway with caveat)
  const handOff = med(samples.map(s => {
    const sh = mid(s.pts[LM.L_SHO], s.pts[LM.R_SHO]), wr = mid(s.pts[LM.L_WRI], s.pts[LM.R_WRI]);
    return (wr.x - sh.x) / ppm * 100;
  }));

  const depthScore = depthMed <= 85 ? 100 : depthMed <= 100 ? 85 : depthMed <= 120 ? 55 : 25;
  const plankScore = hipLine < 0.035 && minBodyAng > 160 ? 100 : hipLine < 0.06 ? 75 : hipLine < 0.1 ? 50 : 25;
  const symScore = clamp01(1 - symmetry / 25) * 100;
  const lockScore = topMed > 165 ? 100 : topMed > 145 ? 70 : 40;
  const tempoScore = tempo === null ? 80 : (tempo > 0.8 && tempo < 5) ? 100 : tempo <= 0.8 ? 55 : 75;

  const subs = {
    "Depth (elbow @ bottom)": Math.round(depthScore),
    "Body line / plank": Math.round(plankScore),
    "Arm lockout at top": Math.round(lockScore),
    "Left/right symmetry": Math.round(symScore),
    "Tempo control": Math.round(tempoScore)
  };
  const score = Math.round(depthScore*.3 + plankScore*.3 + lockScore*.15 + symScore*.15 + tempoScore*.1);

  const repN = Math.max(bottoms.length, 1);
  const totalWork = repN * cfg.bodyweightKg * 9.81 * 0.72 * 0.65; // chest travel ~0.65 of shoulder height factor
  const minutes = (samples.at(-1).t - samples[0].t) * kT / 60;

  const rep = {
    mode: "Push-up form", score, subs,
    metrics: [
      ...trimMetrics(),
      ["Reps detected", String(repN), ""],
      ["Elbow angle @ bottom", `${depthMed.toFixed(0)}°`, depthMed <= 90 ? "full ROM ✓" : "not quite 90°"],
      ["Elbow angle @ top", `${topMed.toFixed(0)}°`, topMed > 165 ? "full lockout ✓" : ""],
      ["Hip line deviation", showIn(hipLine * 100, 1), hipLine < 0.04 ? "rigid plank" : "sag/pike detected"],
      ["Shoulder-hip-ankle angle @ bottom", `${minBodyAng.toFixed(0)}°`, "target 170–180°"],
      ["L/R elbow asymmetry", `${symmetry.toFixed(0)}°`, ""],
      ["Rep tempo", tempo ? `${tempo.toFixed(2)} s/rep` : "n/a", ""],
      ["Hand vs shoulder offset", showIn(handOff, 1), "front-view only — see note"],
      ["Est. total work", `${(totalWork / 1000).toFixed(2)} kJ`, `${(totalWork / 1000 / Math.max(minutes, .01) / 60).toFixed(2)} W avg (if tempo used)`]
    ]
  };

  rep.perspective = pushPerspective(repN, depthMed, cfg);
  rep.feedback = pushFeedback({ depthMed, hipLine, minBodyAng, symmetry, tempo, topMed, handOff, repN, score, cfg });
  return rep;
}

function pushPerspective(reps, depth, cfg) {
  const rows = [];
  const fit = {
    beginner: "average untrained adult manages 5–10 half-depth push-ups",
    recreational: "regular gym-goers: 15–25 full-depth reps",
    advanced: "trained athletes: 30–50+ strict reps",
    elite: "military/elite fitness standards: 60–100+ in a set"
  };
  rows.push({ tier: "info", txt: `Benchmark — ${fit[cfg.level]}. You performed ${reps} detected rep(s)${depth > 90 ? " (not yet full depth)" : " at full depth — the depth is what separates a 'push-up' from a *training* push-up"}.` });
  rows.push({ tier: "info", txt: `Perspective for the average person: push-up capacity is one of the strongest simple markers of cardiovascular & functional fitness on record (a 2019 Harvard study: men doing 40+ push-ups had dramatically lower cardiac-event risk). It's also the safest upper-body exercise you can do anywhere.` });
  rows.push({ tier: "info", txt: `For lifters: full-ROM push-ups build the scapular control and lower-back bracing that carry into bench press and overhead press. Quality reps with a rigid plank beat double the sloppy count.` });
  return rows;
}

function pushFeedback(m) {
  const F = [];
  const add = (level, area, tip) => F.push({ level, area, tip });
  if (m.depthMed > 110) add("bad", "Depth", `You stopped ${Math.round(m.depthMed - 90)}° short of 90° elbow. Lower until your chest is a fist off the floor — half reps build half muscle. If you can't hold depth with a flat back, do incline push-ups on a bench/chair and progress down weekly.`);
  else if (m.depthMed > 90) add("ok", "Depth", `Close — bottom elbow angle ${m.depthMed.toFixed(0)}°, a few degrees shy of parallel-upper-arm. One more centimetre per week and you'll pass the 90° mark.`);
  else add("good", "Depth", `Full depth (${m.depthMed.toFixed(0)}° elbows) — chest to fist-height is exactly right; that's where the pec stretch and growth live.`);

  if (m.hipLine > 0.1) add("bad", "Body line", `Hips dropped ${showIn(m.hipLine * 100, 1)} out of the shoulder-ankle line — that's a 'banana back'. Every sag rep is lumbar spine taking load the arms should. Fix: squeeze glutes HARD, brace abs, and shrink the set — a 6-rep plank-perfect set beats 20 saggy ones. Regress to knee push-ups until the line holds.`);
  else if (m.hipLine > 0.04) add("warn", "Body line", `Slight hip drift (${showIn(m.hipLine * 100, 1)}). Usually the first sign the core is fatiguing — end the set when the drift appears, or do 3 shorter sets.`);
  else add("good", "Body line", `Textbook plank: hip deviation only ${showIn(m.hipLine * 100, 1)}. Ear-shoulder-hip-ankle stayed a straight line — that rigidity is the whole point of the push-up.`);

  if (m.minBodyAng < 150) add("warn", "Body line", `Shoulder-hip-ankle angle closed to ${m.minBodyAng.toFixed(0)}° at the bottom (hip pike or sag). Target a rigid 170–180°.`);

  if (m.topMed < 145) add("ok", "Lockout", `You didn't reach full elbow lockout at the top (${m.topMed.toFixed(0)}°). Push the floor away until arms click straight and shoulders round slightly forward — full lockout completes the serratus work.`);

  if (m.symmetry > 15) add("warn", "Symmetry", `Arms differed by ${m.symmetry.toFixed(0)}° at the bottom — one side is doing more work. Check which side drifts; add slow unilateral-weight (shift chest toward one arm) push-ups, and re-test from the opposite camera side to rule out perspective error.`);
  else add("good", "Symmetry", `Arms worked evenly (${m.symmetry.toFixed(0)}° difference).`);

  if (m.tempo && m.tempo < 0.8) add("warn", "Tempo", `${m.tempo.toFixed(2)} s/rep is a bounce, not a rep. Own the descent: 2 s down, 1 cm pause at bottom, explode up. Slow eccentrics are where most of the strength & hypertrophy signal comes from.`);
  else if (m.tempo) add("good", "Tempo", `Controlled ${m.tempo.toFixed(2)} s tempo — quality cadence.`);

  add("info", "Hand & head position", `Hand offset read ${showIn(m.handOff, 1)} from the shoulder line (only meaningful on a front-view clip; side-view can't judge it). Rule of thumb: hands just outside shoulder width, fingers forward, thumbs near chest-line. Head neutral — eyes on a spot ~12 in ahead of your hands, chin tucked slightly; don't drop the head toward the floor or crane the neck forward.`);
  add("info", "Elbow path", `Keep elbows at ~20–45° from the torso (arrow shape). Flaring to a 'T' loads the shoulder capsule; this camera can check your sagittal plane, film a front clip to audit elbow flare.`);
  if (m.score < 70) add("info", "Programming", `Progression: 4 sets of your perfect-rep count (stop the set the moment form breaks), 3×/week; retest with this app in 3 weeks — add load via backpack once you clear 20 strict reps.`);
  else add("info", "Programming", `Form is solid (${m.score}/100). Next level: deficit push-ups (hands on blocks, +2–3 in depth), 3-sec eccentric, or weighted vest 3×8 — then come back and see the numbers hold.`);
  return F;
}

/* =========================================================
   RENDERING
========================================================= */
function setProgress(p, txt) {
  $("progressWrap").classList.remove("hidden");
  $("progressBar").style.width = (p * 100).toFixed(1) + "%";
  if (txt) $("progressText").textContent = txt;
  if (p >= 1) setTimeout(() => $("progressWrap").classList.add("hidden"), 600);
}
const status = (t, bad) => { statusMsg.textContent = t; statusMsg.style.color = bad ? "#ff6b6b" : ""; };

function renderReport(rep) {
  $("results").classList.remove("hidden");
  $("scoreBig").textContent = rep.score;
  const grade = rep.score >= 90 ? "S — Elite mechanics" : rep.score >= 80 ? "A — Excellent" : rep.score >= 70 ? "B — Good, fine-tune below" : rep.score >= 55 ? "C — Fair, fix priority items" : "D — Rebuild the pattern";
  $("gradeLabel").textContent = `${rep.mode} · ${grade}`;
  $("scoreSummary").textContent = `Composite biomechanics score for the reps in this clip (quality-weighted, not rep count).`;

  $("subscores").innerHTML = Object.entries(rep.subs).map(([k, v]) =>
    `<div class="sub"><span>${k}</span><div class="bar"><div class="fill ${v>=80?'g':v>=60?'y':'r'}" style="width:${v}%"></div></div><b>${v}</b></div>`).join("");

  $("perspective").innerHTML = rep.perspective.map(r =>
    `<p class="persp ${r.tier}">${r.txt}</p>`).join("");

  $("metrics").innerHTML = rep.metrics.map(([k, v, note]) =>
    `<div class="metric"><div class="mk">${k}</div><div class="mv">${v}</div><div class="mn">${note || "&nbsp;"}</div></div>`).join("");

  const ico = { good: "🟢", ok: "🟡", warn: "⚠️", bad: "🔴", info: "💡" };
  $("feedback").innerHTML = rep.feedback.map(f =>
    `<div class="fb ${f.level}"><div class="fbh">${ico[f.level]} <b>${f.area}</b></div><div>${f.tip}</div></div>`).join("");

  $("results").scrollIntoView({ behavior: "smooth" });
}

/* skeleton overlay synced to playback */
function drawOverlayAt(t) {
  if (!samples.length || !srcVideo.videoWidth) return;
  if (overlay.width !== srcVideo.videoWidth) { overlay.width = srcVideo.videoWidth; overlay.height = srcVideo.videoHeight; }
  octx.clearRect(0, 0, overlay.width, overlay.height);
  let best = samples[0];
  for (const s of samples) if (Math.abs(s.t - t) < Math.abs(best.t - t)) best = s;
  const p = best.pts;
  const pairs = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28],[0,11],[0,12],[0,7],[0,8]];
  octx.lineWidth = 3;
  for (const [a, b] of pairs) {
    if (p[a].v < 0.3 || p[b].v < 0.3) continue;
    octx.strokeStyle = (a+b) % 2 ? "#4ade80" : "#38bdf8";
    octx.beginPath(); octx.moveTo(p[a].x, p[a].y); octx.lineTo(p[b].x, p[b].y); octx.stroke();
  }
  for (const pt of p.slice(11)) {
    if (pt.v < 0.3) continue;
    octx.fillStyle = "#fbbf24";
    octx.beginPath(); octx.arc(pt.x, pt.y, 4, 0, 7); octx.fill();
  }
}
srcVideo.addEventListener("timeupdate", () => {
  const w = samplesInfo?.trim;
  // play just the analyzed window: wrap around at its end, never yank a paused scrub
  if (w && !srcVideo.paused && srcVideo.currentTime > w.to) srcVideo.currentTime = w.from;
  overlay.classList.remove("hidden");
  drawOverlayAt(srcVideo.currentTime);
});

$("exportBtn").onclick = () => {
  if (!lastReport) return;
  const r = lastReport;
  const txt = [
    `JumpLab report — ${r.mode} — ${new Date().toLocaleString()}`,
    lastCfg ? `Athlete: ${showLb(lastCfg.bodyweightKg)} BW + ${showLb(lastCfg.loadKg)} load · ${showFtIn(lastCfg.heightCm)} tall · ${lastCfg.level} · ${lastCfg.sport}` : "",
    samplesInfo ? `Sampling: ${samplesInfo.frames} poses @ ~${samplesInfo.effective} fps (source ${samplesInfo.source || "unknown"} fps)` +
      (samplesInfo.trim ? ` · window ${samplesInfo.trim.from.toFixed(2)}-${samplesInfo.trim.to.toFixed(2)} s (${samplesInfo.trim.trimmed.toFixed(1)} s dead time cut, ${samplesInfo.trim.how})` : "") : "",
    `Score: ${r.score}/100`, "",
    "SUB-SCORES", ...Object.entries(r.subs).map(([k, v]) => `  ${k}: ${v}`), "",
    "METRICS", ...r.metrics.map(([k, v, n]) => `  ${k}: ${v} ${n ? "(" + n + ")" : ""}`), "",
    "PERSPECTIVE", ...r.perspective.map(p => "  • " + p.txt), "",
    "COACHING FEEDBACK", ...r.feedback.map(f => `  [${f.level.toUpperCase()}] ${f.area}: ${f.tip}`)
  ].join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([txt], { type: "text/plain" }));
  a.download = "jumplab-report.txt"; a.click();
};

/* =========================================================
   MAIN
========================================================= */
let lastCfg = null;

$("analyzeBtn").onclick = async () => {
  try {
    $("results").classList.add("hidden");
    status("Analyzing…");
    await ensureModel();
    const lb = +$("bodyweightLb").value || 165;
    const loadLb = +$("loadLb").value || 0;
    const cfg = lastCfg = {
      bodyweightKg: lbToKg(lb),
      loadKg: lbToKg(loadLb),
      level: $("level").value,
      heightCm: heightCmFromInputs(),
      sport: $("sport").value,
      view: $("view").value,
      timeScale: 1 / (+($("clipSpeed").value) || 1)   // real seconds per file second
    };
    samples = await sampleVideo(+$("fpsSel").value, activeTab === "pushup" ? "pushup" : "jump", $("trimChk").checked);
    if (!samples.length) throw "No person detected in the video. Use a clip where the whole body is visible and reasonably large in frame.";

    lastReport = activeTab === "pushup" ? analyzePushUp(samples, cfg) : analyzeJump(samples, { ...cfg, view: cfg.view });
    renderReport(lastReport);
    status(`Done. Analyzed ${samples.length} poses at ~${samplesInfo.effective} fps` +
      (samplesInfo.source ? ` (source: ${samplesInfo.source} fps)` : "") +
      (samplesInfo.capped ? ` — your ${samplesInfo.requested} fps choice was capped at the file's own frame rate.` : "") +
      (samplesInfo.trim ? ` Cut ${samplesInfo.trim.trimmed.toFixed(1)} s of dead time (${samplesInfo.trim.how}); playing that window only.` : "") +
      " Scrub the video to see the tracked skeleton overlay.");
    if (samplesInfo.trim) { srcVideo.currentTime = samplesInfo.trim.from; srcVideo.play().catch(() => {}); }
    else srcVideo.play().catch(() => {});
  } catch (e) {
    status(typeof e === "string" ? e : "Analysis failed: " + e.message + " (try reloading the page if the model failed to download)", true);
    console.error(e);
  }
};

initHeightSelects();
