"""Export the falling person's 3D mesh sequence as a self-contained 360-degree
interactive WebGL viewer.

Reuses the falling-person tracker from video_demo.py so only the person who is
falling is reconstructed; everything/everyone else is ignored. For each frame we
run HMR2 to get the SMPL mesh, place it in the camera's 3D frame, lightly smooth
the trajectory, then bake the whole sequence into one standalone HTML file you
can open in any browser and orbit around freely.

Usage:
    python export_mesh_sequence.py \
        --video data/fall_dataset_clips/001_V01/fall_002.mp4 \
        --out_html fall_out/V001_fall_002_viewer.html \
        --frame_stride 2
"""
from pathlib import Path
import argparse
import base64
import json
import os

import cv2
import numpy as np
import torch

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from hmr2.models import DEFAULT_CHECKPOINT, download_models, load_hmr2
from hmr2.utils import recursive_to
from hmr2.utils.renderer import cam_crop_to_full

# Reuse the exact tracking logic already tuned for these fall clips.
from video_demo import (
    build_detector,
    scan_falling_track,
    detect_people,
    match_subject_box,
    smooth_box,
)


def infer_mesh(img_cv2, box, model, model_cfg, device):
    """Return (verts[6890,3], cam_t[3]) for the given subject box, camera frame."""
    dataset = ViTDetDataset(model_cfg, img_cv2, box[None])
    batch = recursive_to(
        next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0))),
        device,
    )
    with torch.no_grad():
        out = model(batch)

    pred_cam = out['pred_cam']
    box_center_t = batch['box_center'].float()
    box_size = batch['box_size'].float()
    img_size = batch['img_size'].float()
    scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
    cam_t_full = cam_crop_to_full(
        pred_cam, box_center_t, box_size, img_size, scaled_focal_length
    ).detach().cpu().numpy()[0]

    verts = out['pred_vertices'][0].detach().cpu().numpy()
    return verts.astype(np.float32), cam_t_full.astype(np.float32)


def collect_sequence(video_path, detector, model, model_cfg, device,
                     frame_stride, max_frames, smooth_alpha, cam_smooth):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video: {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    track = scan_falling_track(cap, detector, frame_stride)
    if track is None:
        raise RuntimeError(f'No falling person found in {video_path}')

    frame_idx = 0
    processed = 0
    prev_box = track[0][1].astype(np.float32)
    smooth_state = prev_box.copy()
    cam_ema = None

    frames_world = []  # each: (6890,3) verts already placed with cam_t
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue
        if max_frames and processed >= max_frames:
            break

        boxes, scores = detect_people(detector, frame)
        matched = match_subject_box(prev_box, boxes, scores)
        smooth_state = smooth_box(smooth_state, matched, alpha=smooth_alpha)
        prev_box = smooth_state

        verts, cam_t = infer_mesh(frame, smooth_state, model, model_cfg, device)

        # Smooth the weak-perspective camera translation so the person does not
        # jitter toward/away from the viewer between independent per-frame fits.
        if cam_ema is None:
            cam_ema = cam_t
        else:
            cam_ema = cam_smooth * cam_t + (1.0 - cam_smooth) * cam_ema

        frames_world.append(verts + cam_ema[None, :])
        processed += 1
        frame_idx += 1
        print(f'{Path(video_path).stem}: mesh frame {processed}', flush=True)

    cap.release()
    if not frames_world:
        raise RuntimeError('No frames reconstructed')

    out_fps = max(fps / frame_stride, 1.0)
    return np.stack(frames_world, axis=0), out_fps  # (F,6890,3), fps


def to_y_up(seq):
    """HMR2 verts are Y-down / Z-forward (image convention). Convert to Y-up."""
    seq = seq.copy()
    seq[..., 1] *= -1.0
    seq[..., 2] *= -1.0
    return seq


def find_grounded_frame(seq):
    """Pick the frame where the falling person is settled on the ground."""
    foot_y = seq[:, :, 1].min(axis=1)
    tail_start = max(len(foot_y) // 3, 0)

    # Prefer the most horizontal body pose in the latter part of the clip
    # (works for both world-space and pelvis-centered exports).
    best_f = tail_start
    best_score = -1e9
    for f in range(tail_start, len(foot_y)):
        pts = seq[f]
        y_ext = float(np.ptp(pts[:, 1]))
        h_ext = float(np.ptp(pts[:, 0]) + np.ptp(pts[:, 2]))
        flatness = h_ext / (y_ext + 1e-6)
        score = flatness - 0.05 * y_ext
        if score > best_score:
            best_score = score
            best_f = f
    return int(best_f)


def align_for_ground_view(seq, grounded_frame):
    """Put the ground at y=0 and center XY on the fallen person for 360 orbit."""
    seq = seq.copy()
    ground_y = float(seq[grounded_frame, :, 1].min())
    seq[:, :, 1] -= ground_y
    center = seq[grounded_frame].mean(axis=0)
    seq[:, :, 0] -= center[0]
    seq[:, :, 2] -= center[2]
    return seq


def quantize(seq):
    """Quantize all vertex positions to int16 with a shared scale/offset."""
    flat = seq.reshape(-1, 3)
    lo = flat.min(axis=0)
    hi = flat.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    # map to [-32767, 32767]
    norm = (seq - lo[None, None, :]) / span[None, None, :]  # 0..1
    q = np.round(norm * 65534.0 - 32767.0).astype(np.int16)
    return q, lo.astype(np.float32), span.astype(np.float32)


def build_html(seq_q, lo, span, faces, fps, meta, grounded_frame, video_src=None, camera=None):
    F, V, _ = seq_q.shape
    pos_b64 = base64.b64encode(seq_q.tobytes()).decode('ascii')
    faces_u16 = faces.astype(np.uint16)
    faces_b64 = base64.b64encode(faces_u16.tobytes()).decode('ascii')

    cfg = {
        'numFrames': int(F),
        'numVerts': int(V),
        'numFaces': int(faces.shape[0]),
        'lo': [float(x) for x in lo],
        'span': [float(x) for x in span],
        'fps': float(fps),
        'groundedFrame': int(grounded_frame),
        'videoSrc': video_src,
        'camera': camera,
        'meta': meta,
    }
    cfg_json = json.dumps(cfg)

    html = _HTML_TEMPLATE
    html = html.replace('/*__CONFIG__*/', cfg_json)
    html = html.replace('__POS_B64__', pos_b64)
    html = html.replace('__FACES_B64__', faces_b64)
    html = html.replace('__TITLE__', meta.get('title', 'Fall Mesh Viewer'))
    return html


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { color-scheme: dark; }
  html,body { margin:0; height:100%; background:#000; color:#e6edf3;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; overflow:hidden; }
  #bgVideo { position:fixed; inset:0; width:100%; height:100%; object-fit:contain; background:#000; z-index:0; display:none; }
  #glcanvas { position:fixed; inset:0; width:100%; height:100%; display:block; z-index:1; cursor:grab; }
  #glcanvas:active { cursor:grabbing; }
  #panel { position:fixed; top:12px; left:12px; background:rgba(17,22,32,.86);
    border:1px solid #223; border-radius:12px; padding:14px 16px; width:270px;
    backdrop-filter:blur(8px); box-shadow:0 8px 30px rgba(0,0,0,.5); }
  #panel h1 { font-size:14px; margin:0 0 2px; font-weight:650; }
  #panel .sub { font-size:11px; color:#8b98a9; margin:0 0 12px; }
  .row { display:flex; align-items:center; gap:8px; margin:8px 0; font-size:12px; }
  .row label { flex:0 0 auto; color:#aab6c6; }
  input[type=range] { flex:1; accent-color:#4f9cff; }
  button { background:#1b2434; color:#e6edf3; border:1px solid #2c3a52;
    border-radius:8px; padding:6px 10px; font-size:12px; cursor:pointer; }
  button:hover { background:#243149; }
  button.on { background:#2b5fa8; border-color:#4f9cff; }
  .btns { display:flex; flex-wrap:wrap; gap:6px; margin-top:6px; }
  #frameLabel { font-variant-numeric:tabular-nums; color:#8b98a9; font-size:11px; }
  #hint { position:fixed; bottom:10px; left:12px; font-size:11px; color:#66738a; }
  kbd { background:#1b2434; border:1px solid #2c3a52; border-radius:4px; padding:1px 5px; font-size:10px; }
</style>
</head>
<body>
<video id="bgVideo" muted playsinline></video>
<canvas id="glcanvas"></canvas>
<div id="panel">
  <h1>__TITLE__</h1>
  <p class="sub" id="metaSub"></p>
  <div class="row">
    <button id="playBtn" class="on">Pause</button>
    <span id="frameLabel">0 / 0</span>
  </div>
  <div class="row">
    <label>Frame</label>
    <input id="scrub" type="range" min="0" max="0" value="0" step="1">
  </div>
  <div class="row">
    <label>Speed</label>
    <input id="speed" type="range" min="0.1" max="3" value="1" step="0.1">
    <span id="speedLabel" style="font-size:11px;color:#8b98a9">1.0x</span>
  </div>
  <div class="btns">
    <button id="view360Btn">360° Inspect</button>
    <button id="spinBtn">Auto-rotate</button>
    <button id="resetBtn">Reset view</button>
  </div>
  <div class="btns">
    <button id="wireBtn">Wireframe</button>
    <button id="gridBtn" class="on">Grid</button>
  </div>
</div>
<div id="hint"><b>Play</b> = mesh on video (same as overlay.mp4) &nbsp;·&nbsp; <b>360° Inspect</b> or <kbd>Space</kbd> pause → drag to orbit mesh</div>

<script>
const CONFIG = /*__CONFIG__*/;
const POS_B64 = "__POS_B64__";
const FACES_B64 = "__FACES_B64__";

function b64ToBytes(b64){
  const bin = atob(b64); const len = bin.length;
  const bytes = new Uint8Array(len);
  for (let i=0;i<len;i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}
// Dequantize positions: int16 -> float
const qpos = new Int16Array(b64ToBytes(POS_B64).buffer);
const facesU16 = new Uint16Array(b64ToBytes(FACES_B64).buffer);
const F = CONFIG.numFrames, V = CONFIG.numVerts, NF = CONFIG.numFaces;
const lo = CONFIG.lo, span = CONFIG.span;

// Precompute all frames as Float32 vertex arrays (dequantized), Y-up world.
const framesPos = [];
for (let f=0; f<F; f++){
  const arr = new Float32Array(V*3);
  const base = f*V*3;
  for (let i=0;i<V;i++){
    for (let c=0;c<3;c++){
      const q = qpos[base + i*3 + c];
      arr[i*3+c] = ((q + 32767) / 65534) * span[c] + lo[c];
    }
  }
  framesPos.push(arr);
}
// Per-frame centroids + centered verts for 360° inspect mode
const centroids = [];
const centeredFrames = [];
const frameRadii = [];
const frameMinY = [];
for (let f=0; f<F; f++){
  const p = framesPos[f]; let cx=0,cy=0,cz=0;
  const carr = new Float32Array(V*3);
  let minY=1e9, maxR=0;
  for (let i=0;i<V;i++){ cx+=p[i*3]; cy+=p[i*3+1]; cz+=p[i*3+2]; }
  cx/=V; cy/=V; cz/=V;
  centroids.push([cx,cy,cz]);
  for (let i=0;i<V;i++){
    const x=p[i*3]-cx, y=p[i*3+1]-cy, z=p[i*3+2]-cz;
    carr[i*3]=x; carr[i*3+1]=y; carr[i*3+2]=z;
    minY=Math.min(minY,y); maxR=Math.max(maxR,Math.hypot(x,y,z));
  }
  centeredFrames.push(carr);
  frameRadii.push(maxR);
  frameMinY.push(minY);
}
// Overall bounds for camera fit + grid placement
let minY=1e9, maxR=0, cAll=[0,0,0];
for (let f=0; f<F; f++){ cAll[0]+=centroids[f][0]; cAll[1]+=centroids[f][1]; cAll[2]+=centroids[f][2]; }
cAll = cAll.map(v=>v/F);
for (let f=0; f<F; f++){ const p=framesPos[f];
  for (let i=0;i<V;i++){ const y=p[i*3+1]; if(y<minY)minY=y;
    const dx=p[i*3]-cAll[0], dy=p[i*3+1]-cAll[1], dz=p[i*3+2]-cAll[2];
    const r=Math.hypot(dx,dy,dz); if(r>maxR)maxR=r; } }

// ---- WebGL setup ----
const hasVideo = !!CONFIG.videoSrc;
const bgVideo = document.getElementById('bgVideo');
if (hasVideo) {
  bgVideo.src = CONFIG.videoSrc;
  bgVideo.style.display = 'block';
}
const canvas = document.getElementById('glcanvas');
const gl = canvas.getContext('webgl', {antialias:true, alpha:hasVideo});
if(!gl){ document.body.innerHTML='<p style="padding:2em">WebGL not available in this browser.</p>'; }

function syncVideo(){
  if(!hasVideo || !bgVideo.duration) return;
  const t = Math.min(frame / CONFIG.fps, Math.max(bgVideo.duration - 0.001, 0));
  if(Math.abs(bgVideo.currentTime - t) > 0.02) bgVideo.currentTime = t;
}

function compile(type, src){ const s=gl.createShader(type); gl.shaderSource(s,src); gl.compileShader(s);
  if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)) throw gl.getShaderInfoLog(s); return s; }
function program(vs, fs){ const p=gl.createProgram(); gl.attachShader(p,compile(gl.VERTEX_SHADER,vs));
  gl.attachShader(p,compile(gl.FRAGMENT_SHADER,fs)); gl.linkProgram(p);
  if(!gl.getProgramParameter(p,gl.LINK_STATUS)) throw gl.getProgramInfoLog(p); return p; }

const meshProg = program(
 `attribute vec3 aPos; attribute vec3 aNormal;
  uniform mat4 uMVP; uniform mat4 uModel;
  varying vec3 vN; varying vec3 vWorld;
  void main(){ vN = mat3(uModel)*aNormal; vWorld=(uModel*vec4(aPos,1.0)).xyz;
    gl_Position = uMVP*vec4(aPos,1.0); }`,
 `precision highp float; varying vec3 vN; varying vec3 vWorld;
  uniform vec3 uColor; uniform float uWire;
  void main(){ vec3 N=normalize(vN);
    vec3 L1=normalize(vec3(0.4,0.8,0.6)); vec3 L2=normalize(vec3(-0.5,0.3,-0.7));
    float d = 0.30 + 0.72*max(dot(N,L1),0.0) + 0.25*max(dot(N,L2),0.0);
    vec3 col = uColor*d;
    if(uWire>0.5) col = vec3(0.55,0.78,1.0);
    gl_FragColor = vec4(col,1.0); }`
);
const gridProg = program(
 `attribute vec3 aPos; uniform mat4 uMVP; void main(){ gl_Position=uMVP*vec4(aPos,1.0); }`,
 `precision mediump float; uniform vec3 uColor; void main(){ gl_FragColor=vec4(uColor,1.0); }`
);

// Buffers
const posBuf = gl.createBuffer();
const normBuf = gl.createBuffer();
const idxBuf = gl.createBuffer();
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, facesU16, gl.STATIC_DRAW);

// Wireframe index buffer (edges)
const edges = new Uint16Array(NF*6);
for(let i=0;i<NF;i++){ const a=facesU16[i*3],b=facesU16[i*3+1],c=facesU16[i*3+2];
  edges[i*6]=a;edges[i*6+1]=b;edges[i*6+2]=b;edges[i*6+3]=c;edges[i*6+4]=c;edges[i*6+5]=a; }
const edgeBuf = gl.createBuffer();
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, edgeBuf);
gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, edges, gl.STATIC_DRAW);

// Ground grid geometry
function makeGrid(y, half, step){
  const v=[]; for(let x=-half;x<=half;x+=step){ v.push(x,y,-half, x,y,half); v.push(-half,y,x, half,y,x); }
  return new Float32Array(v);
}
const gridVerts = makeGrid(minY - 0.01, Math.max(1.2, maxR*2.2), Math.max(0.15, maxR*2.2/24));
const gridBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, gridBuf); gl.bufferData(gl.ARRAY_BUFFER, gridVerts, gl.STATIC_DRAW);
const gridCount = gridVerts.length/3;

// Normals cache per uploaded frame
function computeNormals(pos){
  const n = new Float32Array(V*3);
  for(let i=0;i<NF;i++){
    const ia=facesU16[i*3]*3, ib=facesU16[i*3+1]*3, ic=facesU16[i*3+2]*3;
    const ax=pos[ia],ay=pos[ia+1],az=pos[ia+2];
    const bx=pos[ib],by=pos[ib+1],bz=pos[ib+2];
    const cx=pos[ic],cy=pos[ic+1],cz=pos[ic+2];
    const e1x=bx-ax,e1y=by-ay,e1z=bz-az;
    const e2x=cx-ax,e2y=cy-ay,e2z=cz-az;
    const nx=e1y*e2z-e1z*e2y, ny=e1z*e2x-e1x*e2z, nz=e1x*e2y-e1y*e2x;
    n[ia]+=nx;n[ia+1]+=ny;n[ia+2]+=nz;
    n[ib]+=nx;n[ib+1]+=ny;n[ib+2]+=nz;
    n[ic]+=nx;n[ic+1]+=ny;n[ic+2]+=nz;
  }
  for(let i=0;i<V;i++){ const x=n[i*3],y=n[i*3+1],z=n[i*3+2]; const l=Math.hypot(x,y,z)||1;
    n[i*3]=x/l;n[i*3+1]=y/l;n[i*3+2]=z/l; }
  return n;
}

// ---- Minimal mat4 ----
function m4id(){return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];}
function m4mul(a,b){const o=new Array(16);
  for(let r=0;r<4;r++)for(let c=0;c<4;c++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+r]*b[c*4+k];o[c*4+r]=s;}return o;}
function m4persp(fov,asp,n,f){const t=1/Math.tan(fov/2);
  return [t/asp,0,0,0, 0,t,0,0, 0,0,(f+n)/(n-f),-1, 0,0,(2*f*n)/(n-f),0];}
function m4translate(x,y,z){const m=m4id();m[12]=x;m[13]=y;m[14]=z;return m;}
function m4lookAt(eye,ctr,up){
  let zx=eye[0]-ctr[0],zy=eye[1]-ctr[1],zz=eye[2]-ctr[2];let zl=Math.hypot(zx,zy,zz)||1;zx/=zl;zy/=zl;zz/=zl;
  let xx=up[1]*zz-up[2]*zy,xy=up[2]*zx-up[0]*zz,xz=up[0]*zy-up[1]*zx;let xl=Math.hypot(xx,xy,xz)||1;xx/=xl;xy/=xl;xz/=xl;
  let yx=zy*xz-zz*xy,yy=zz*xx-zx*xz,yz=zx*xy-zy*xx;
  return [xx,yx,zx,0, xy,yy,zy,0, xz,yz,zz,0,
    -(xx*eye[0]+xy*eye[1]+xz*eye[2]), -(yx*eye[0]+yy*eye[1]+yz*eye[2]), -(zx*eye[0]+zy*eye[1]+zz*eye[2]), 1];}

// ---- Camera / orbit state ----
const GROUND = CONFIG.groundedFrame || 0;
let inspectMode = false;
let yaw=0.4, pitch=0.22, dist=maxR*2.8, target=[0, maxR*0.15, 0];
const defaultCam = {yaw,pitch,dist,target:[...target]};
function resetView(){
  yaw=defaultCam.yaw; pitch=defaultCam.pitch;
  dist=Math.max(frameRadii[frame]*2.8,0.8);
  target=inspectMode ? [0, frameRadii[frame]*0.15, 0] : [...defaultCam.target];
}
function enterInspect(){
  inspectMode = true;
  if(hasVideo) bgVideo.style.opacity = '0';
  showGrid = true; gridBtn.classList.add('on');
  dist = Math.max(frameRadii[frame]*2.8, 0.8);
  target = [0, frameRadii[frame]*0.15, 0];
  yaw = 0.4; pitch = 0.22;
  uploaded = -1;
  canvas.style.cursor = 'grab';
}
function exitInspect(){
  inspectMode = false;
  if(hasVideo) bgVideo.style.opacity = '1';
  showGrid = false; gridBtn.classList.remove('on');
  uploaded = -1;
  canvas.style.cursor = 'default';
}
function getProjection(){
  if(hasVideo && playing && !inspectMode && CONFIG.camera){
    const cam = CONFIG.camera;
    const fovY = 2*Math.atan(cam.height / (2*cam.focal));
    return m4persp(fovY, canvas.width/canvas.height, 0.05, 100);
  }
  const r = inspectMode ? frameRadii[frame] : maxR;
  return m4persp(50*Math.PI/180, canvas.width/canvas.height, r*0.02, r*50);
}
function getView(){
  if(hasVideo && playing && !inspectMode) return m4id();
  const eye=[ target[0]+dist*Math.cos(pitch)*Math.sin(yaw),
             target[1]+dist*Math.sin(pitch),
             target[2]+dist*Math.cos(pitch)*Math.cos(yaw) ];
  return m4lookAt(eye,target,[0,1,0]);
}

let dragging=false, panning=false, lastX=0, lastY=0;
canvas.addEventListener('mousedown',e=>{
  if(playing) return; // orbit only while paused
  dragging=true; panning=(e.button===2); lastX=e.clientX; lastY=e.clientY;
});
window.addEventListener('mouseup',()=>{dragging=false;panning=false;});
window.addEventListener('mousemove',e=>{ if(!dragging)return;
  const dx=e.clientX-lastX, dy=e.clientY-lastY; lastX=e.clientX; lastY=e.clientY;
  if(panning){ const s=dist*0.0016; const right=[Math.cos(yaw),0,-Math.sin(yaw)];
    const up=[0,1,0]; target[0]-=(right[0]*dx)*s; target[2]-=(right[2]*dx)*s; target[1]+=up[1]*dy*s; }
  else { yaw+=dx*0.01; pitch+=dy*0.01; pitch=Math.max(-1.5,Math.min(1.5,pitch)); }
});
canvas.addEventListener('contextmenu',e=>e.preventDefault());
canvas.addEventListener('wheel',e=>{ if(playing && !inspectMode) return;
  e.preventDefault(); dist*=Math.pow(1.0015,e.deltaY);
  const r = inspectMode ? frameRadii[frame] : maxR;
  dist=Math.max(r*0.35,Math.min(r*12,dist)); },{passive:false});
// touch
let pinchD=0;
canvas.addEventListener('touchstart',e=>{
  if(playing) return;
  if(e.touches.length===1){dragging=true;lastX=e.touches[0].clientX;lastY=e.touches[0].clientY;}
  else if(e.touches.length===2){pinchD=Math.hypot(e.touches[0].clientX-e.touches[1].clientX,e.touches[0].clientY-e.touches[1].clientY);}
},{passive:true});
canvas.addEventListener('touchmove',e=>{ if(e.touches.length===1&&dragging){const dx=e.touches[0].clientX-lastX,dy=e.touches[0].clientY-lastY;
  lastX=e.touches[0].clientX;lastY=e.touches[0].clientY; yaw+=dx*0.01;pitch=Math.max(-1.5,Math.min(1.5,pitch+dy*0.01));}
  else if(e.touches.length===2){const d=Math.hypot(e.touches[0].clientX-e.touches[1].clientX,e.touches[0].clientY-e.touches[1].clientY);
  if(pinchD)dist=Math.max(maxR*0.4,Math.min(maxR*12,dist*pinchD/d)); pinchD=d;} },{passive:true});
canvas.addEventListener('touchend',()=>{dragging=false;pinchD=0;});

// ---- UI ----
let frame=0, playing=true, speed=1.0, spinning=false, wire=false, showGrid=!hasVideo, lockPlace=!hasVideo;
let acc=0;
const playBtn=document.getElementById('playBtn'), scrub=document.getElementById('scrub'),
  speedEl=document.getElementById('speed'), speedLabel=document.getElementById('speedLabel'),
  frameLabel=document.getElementById('frameLabel'), spinBtn=document.getElementById('spinBtn'),
  lockBtn=document.getElementById('lockBtn'), wireBtn=document.getElementById('wireBtn'),
  gridBtn=document.getElementById('gridBtn'), resetBtn=document.getElementById('resetBtn'),
  view360Btn=document.getElementById('view360Btn'), metaSub=document.getElementById('metaSub');
scrub.max=F-1;
const clipDur = (F / CONFIG.fps).toFixed(2);
metaSub.textContent = (CONFIG.meta.source||'') + ' · ' + clipDur + 's · ' + F + ' frames · pause = 360° mesh';
function setPlaying(p){
  playing=p;
  playBtn.textContent=p?'Pause':'Play';
  playBtn.classList.toggle('on',p);
  if(hasVideo){ if(p){ exitInspect(); view360Btn.classList.remove('on'); }
    else { enterInspect(); view360Btn.classList.add('on'); } }
  else { canvas.style.cursor = p ? 'default' : 'grab'; if(!p){ lockPlace=true; if(lockBtn) lockBtn.classList.add('on'); } }
}
function go360(){ setPlaying(false); }
playBtn.onclick=()=>setPlaying(!playing);
if(view360Btn) view360Btn.onclick=go360;
scrub.oninput=()=>{ frame=+scrub.value; if(inspectMode){ dist=Math.max(frameRadii[frame]*2.8,0.8); target=[0,frameRadii[frame]*0.15,0]; uploaded=-1; } setPlaying(false); };
speedEl.oninput=()=>{ speed=+speedEl.value; speedLabel.textContent=speed.toFixed(1)+'x'; };
spinBtn.onclick=()=>{ spinning=!spinning; spinBtn.classList.toggle('on',spinning); };
lockBtn.onclick=()=>{ if(lockBtn){ lockPlace=!lockPlace; lockBtn.classList.toggle('on',lockPlace); } };
wireBtn.onclick=()=>{ wire=!wire; wireBtn.classList.toggle('on',wire); };
gridBtn.onclick=()=>{ showGrid=!showGrid; gridBtn.classList.toggle('on',showGrid); };
resetBtn.onclick=resetView;
window.addEventListener('keydown',e=>{ if(e.code==='Space'){e.preventDefault();setPlaying(!playing);}
  if(e.key==='ArrowRight'){frame=Math.min(F-1,frame+1); if(inspectMode){dist=Math.max(frameRadii[frame]*2.8,0.8);target=[0,frameRadii[frame]*0.15,0];uploaded=-1;} setPlaying(false);}
  if(e.key==='ArrowLeft'){frame=Math.max(0,frame-1); if(inspectMode){dist=Math.max(frameRadii[frame]*2.8,0.8);target=[0,frameRadii[frame]*0.15,0];uploaded=-1;} setPlaying(false);} });

const normCache = new Array(F).fill(null);
const normCacheC = new Array(F).fill(null);
function uploadFrame(f){
  const useCentered = inspectMode || (!hasVideo && !playing);
  const pos = useCentered ? centeredFrames[f] : framesPos[f];
  const nc = useCentered ? normCacheC : normCache;
  if(!nc[f]) nc[f]=computeNormals(pos);
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, pos, gl.DYNAMIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, normBuf);
  gl.bufferData(gl.ARRAY_BUFFER, nc[f], gl.DYNAMIC_DRAW);
}
let uploaded=-1;

function resize(){ const dpr=Math.min(2,window.devicePixelRatio||1);
  canvas.width=canvas.clientWidth*dpr; canvas.height=canvas.clientHeight*dpr;
  gl.viewport(0,0,canvas.width,canvas.height); }
window.addEventListener('resize',resize); resize();

gl.enable(gl.DEPTH_TEST);
if(hasVideo){ gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); }
let lastT=performance.now();
function render(now){
  const dt=(now-lastT)/1000; lastT=now;
  if(playing){
    acc+=dt*speed*CONFIG.fps;
    while(acc>=1){
      if(frame>=F-1){ frame=F-1; setPlaying(false); acc=0; break; }
      frame++; acc-=1;
    }
    if(hasVideo && bgVideo.paused) bgVideo.play().catch(()=>{});
  } else if(hasVideo) {
    bgVideo.pause();
    syncVideo();
  }
  if(spinning && !playing) yaw+=dt*0.6;
  if(hasVideo && playing) syncVideo();
  scrub.value=frame; frameLabel.textContent=(frame+1)+' / '+F;

  if(uploaded!==frame){ uploadFrame(frame); uploaded=frame; }

  let model=m4id();
  if(lockPlace && !hasVideo && !inspectMode){ const c=centroids[frame];
    model=m4translate(cAll[0]-c[0], cAll[1]-c[1], cAll[2]-c[2]); }

  const view=getView();
  const proj=getProjection();
  const vp=m4mul(proj,view);
  const mvp=m4mul(vp,model);

  if(hasVideo && playing && !inspectMode) gl.clearColor(0,0,0,0);
  else gl.clearColor(0.043,0.055,0.078,1);
  gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);

  if(showGrid && (inspectMode || !hasVideo)){
    gl.useProgram(gridProg);
    gl.uniformMatrix4fv(gl.getUniformLocation(gridProg,'uMVP'),false,new Float32Array(vp));
    gl.uniform3f(gl.getUniformLocation(gridProg,'uColor'),0.16,0.20,0.28);
    const gp=gl.getAttribLocation(gridProg,'aPos');
    if(inspectMode){
      const gy = frameMinY[frame] - 0.02;
      const gh = Math.max(0.8, frameRadii[frame]*2.2);
      const gv = makeGrid(gy, gh, Math.max(0.12, gh/24));
      gl.bindBuffer(gl.ARRAY_BUFFER, gridBuf); gl.bufferData(gl.ARRAY_BUFFER, gv, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(gp); gl.vertexAttribPointer(gp,3,gl.FLOAT,false,0,0);
      gl.drawArrays(gl.LINES,0,gv.length/3);
    } else {
      gl.bindBuffer(gl.ARRAY_BUFFER,gridBuf);
      gl.enableVertexAttribArray(gp); gl.vertexAttribPointer(gp,3,gl.FLOAT,false,0,0);
      gl.drawArrays(gl.LINES,0,gridCount);
    }
  }

  // mesh
  gl.useProgram(meshProg);
  gl.uniformMatrix4fv(gl.getUniformLocation(meshProg,'uMVP'),false,new Float32Array(mvp));
  gl.uniformMatrix4fv(gl.getUniformLocation(meshProg,'uModel'),false,new Float32Array(model));
  gl.uniform3f(gl.getUniformLocation(meshProg,'uColor'),0.65,0.74,0.86);
  gl.uniform1f(gl.getUniformLocation(meshProg,'uWire'),wire?1:0);
  const ap=gl.getAttribLocation(meshProg,'aPos'); gl.bindBuffer(gl.ARRAY_BUFFER,posBuf);
  gl.enableVertexAttribArray(ap); gl.vertexAttribPointer(ap,3,gl.FLOAT,false,0,0);
  const an=gl.getAttribLocation(meshProg,'aNormal'); gl.bindBuffer(gl.ARRAY_BUFFER,normBuf);
  gl.enableVertexAttribArray(an); gl.vertexAttribPointer(an,3,gl.FLOAT,false,0,0);
  if(wire){ gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,edgeBuf); gl.drawElements(gl.LINES,NF*6,gl.UNSIGNED_SHORT,0); }
  else { gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,idxBuf); gl.drawElements(gl.TRIANGLES,NF*3,gl.UNSIGNED_SHORT,0); }

  requestAnimationFrame(render);
}
setPlaying(true);
requestAnimationFrame(render);
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description='Export falling-person mesh as a 360 interactive viewer')
    ap.add_argument('--video', required=True)
    ap.add_argument('--out_html', default='fall_out/mesh_viewer.html')
    ap.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    ap.add_argument('--detector', default='regnety', choices=['vitdet', 'regnety'])
    ap.add_argument('--frame_stride', type=int, default=2, help='Lower = smoother, slower')
    ap.add_argument('--max_frames', type=int, default=0)
    ap.add_argument('--smooth_alpha', type=float, default=0.55)
    ap.add_argument('--cam_smooth', type=float, default=0.5, help='Trajectory smoothing (0..1)')
    ap.add_argument('--title', default='Falling Person – 3D Mesh')
    args = ap.parse_args()

    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(args.checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    detector = build_detector(args.detector)

    print('Reconstructing falling-person mesh sequence...', flush=True)
    seq, fps = collect_sequence(
        args.video, detector, model, model_cfg, device,
        args.frame_stride, args.max_frames, args.smooth_alpha, args.cam_smooth,
    )
    print(f'Collected {seq.shape[0]} frames.', flush=True)

    seq = to_y_up(seq)
    grounded_frame = find_grounded_frame(seq)
    seq = align_for_ground_view(seq, grounded_frame)
    seq_q, lo, span = quantize(seq)
    faces = np.asarray(model.smpl.faces)

    meta = {
        'title': args.title,
        'source': Path(args.video).name,
        'fps': fps,
        'grounded_frame': grounded_frame,
        'note': 'Falling person only; play clip then pause to orbit',
    }
    html = build_html(seq_q, lo, span, faces, fps, meta, grounded_frame)

    out_path = Path(args.out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    size_mb = out_path.stat().st_size / 1e6
    print(f'Grounded pose frame: {grounded_frame} / {seq.shape[0] - 1}', flush=True)
    print(f'Saved interactive 360 viewer: {out_path}  ({size_mb:.1f} MB)', flush=True)
    print('Open in a browser — plays at clip speed; pause any frame and drag to orbit.', flush=True)


if __name__ == '__main__':
    main()
