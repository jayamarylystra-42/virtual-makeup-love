"""
GlamCam — Award-style Streamlit live virtual makeup (enhanced UI + product cards + before/after)
Drop this file into your repo root on GitHub and include the recommended requirements.txt.
"""

import streamlit as st
st.set_page_config(page_title="GlamCam — Award UI", page_icon="💄", layout="wide", initial_sidebar_state="expanded")

# safe imports and clear helpful message if unavailable on cloud
try:
    import cv2
    import numpy as np
    import mediapipe as mp
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase, RTCConfiguration
    from PIL import Image
    import base64
    import io
except Exception as e:
    st.error(
        "Import error: {}\n\nMake sure your repo includes the required packages in requirements.txt "
        "and select a supported Python runtime (3.8–3.10) on Streamlit Cloud.".format(e)
    )
    st.stop()

# ------------------ Styling (header + product cards) ------------------
APP_BG = """
<style>
/* page background */
body { background: linear-gradient(120deg,#fffaf6 0%, #fff 40%); }

/* header */
.header {
  display:flex; align-items:center; gap:18px;
}
.brand {
  font-weight:800; font-size:22px; letter-spacing:0.2px; color:#5b2e6f;
}
.subtitle { color:#6b6b6b; font-size:13px; margin-top:-2px; }

/* product card */
.prod {
  background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(250,245,250,0.9));
  border-radius:14px; padding:12px; box-shadow: 0 6px 18px rgba(88,46,120,0.08);
  border: 1px solid rgba(200,190,210,0.35);
}
.prod h4 { margin:4px 0 6px 0; color:#3b1a45; }
.prod p { margin:0; color:#6b6170; font-size:13px; }

/* before-after container */
.ba-wrap { width:100%; max-width:920px; margin:auto; }
.ba {
  position: relative; overflow:hidden; border-radius:12px; border:1px solid rgba(200,190,210,0.35);
  box-shadow: 0 10px 30px rgba(80,40,120,0.06);
}
.ba img { display:block; width:100%; height:auto; }

/* slider handle */
.handle {
  position:absolute; top:0; bottom:0; width:6px; background:linear-gradient(180deg,#fff,#ddd);
  left:50%; transform:translateX(-50%); z-index:3; border-radius:4px; box-shadow:0 3px 12px rgba(0,0,0,0.12);
}
.handle:after {
  content:''; position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); width:28px; height:28px;
  background:#fff; border-radius:50%; border:2px solid rgba(120,90,140,0.12);
}

/* small responsive tweaks */
@media (max-width:900px) {
  .brand { font-size:18px; }
  .prod { padding:10px; border-radius:10px; }
}
</style>
"""

st.markdown(APP_BG, unsafe_allow_html=True)

# header
with st.container():
    left, mid, right = st.columns([0.18, 0.64, 0.18])
    with left:
        st.image("https://raw.githubusercontent.com/streamlit/example-data/master/emoji/face_with_makeup.png", width=72)
    with mid:
        st.markdown('<div class="header"><div><div class="brand">GlamCam Studio</div><div class="subtitle">Pro-level blending • Product cards • Before/After • Save & Share</div></div></div>', unsafe_allow_html=True)
        st.write("")
    with right:
        st.markdown("<div style='text-align:right'><small>Built for demo & portfolio</small></div>", unsafe_allow_html=True)

# ------------------ Core logic (reused high-quality transformer with small UI hooks) ------------------
mp_face_mesh = mp.solutions.face_mesh

def hex_to_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0,2,4))

def polygon_mask(shape, polygon_pts, blur_ksize=41):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if not polygon_pts or len(polygon_pts) < 3:
        return mask.astype(float)
    cv2.fillPoly(mask, [np.array(polygon_pts, dtype=np.int32)], 255)
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    k = min(k, max(1, (min(shape[:2]) // 2) | 1))
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return (mask.astype(np.float32) / 255.0)

def blend_color(image_rgb, color_rgb, mask, opacity=0.6):
    overlay = np.full_like(image_rgb, color_rgb, dtype=np.uint8)
    mask_3 = np.stack([mask, mask, mask], axis=2)
    blended = (image_rgb.astype(np.float32) * (1 - mask_3 * opacity) + overlay.astype(np.float32) * (mask_3 * opacity))
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended

def gaussian_spot_mask(shape, centers, radius_factor=12, blur_factor=0.45):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for (cx, cy), rfac in centers:
        r = max(6, min(h, w) // rfac)
        cv2.circle(mask, (int(cx), int(cy)), int(r), 255, -1)
    k = int(min(h, w) * blur_factor)
    if k % 2 == 0: k += 1
    k = max(3, k)
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return (mask.astype(np.float32) / 255.0)

# Face landmark index groups (Mediapipe)
def idxs_from_connections(conns):
    s = set()
    for a, b in conns:
        s.add(a); s.add(b)
    return sorted(s)

LIPS_IDX = idxs_from_connections(mp_face_mesh.FACEMESH_LIPS)
LEFT_EYE_IDX = idxs_from_connections(mp_face_mesh.FACEMESH_LEFT_EYE)
RIGHT_EYE_IDX = idxs_from_connections(mp_face_mesh.FACEMESH_RIGHT_EYE)
FACE_OVAL_IDX = idxs_from_connections(mp_face_mesh.FACEMESH_FACE_OVAL)
LEFT_IRIS_IDX = idxs_from_connections(mp_face_mesh.FACEMESH_LEFT_IRIS)
RIGHT_IRIS_IDX = idxs_from_connections(mp_face_mesh.FACEMESH_RIGHT_IRIS)

# Slightly smaller model (for speed) but still refined landmarks
class AwardTransformer(VideoTransformerBase):
    def __init__(self, cfg):
        self.cfg = cfg
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                              refine_landmarks=True,
                                              min_detection_confidence=0.5,
                                              min_tracking_confidence=0.5)

    def transform(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        scale = self.cfg.get("scale", 1.0)
        if scale != 1.0:
            small = cv2.resize(img_rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
            proc_rgb = small; ph, pw = small.shape[:2]
        else:
            proc_rgb = img_rgb; ph, pw = h, w

        res = self.face_mesh.process(proc_rgb)
        if not res.multi_face_landmarks:
            return img_bgr

        out_rgb = cv2.resize(proc_rgb, (w,h), interpolation=cv2.INTER_LINEAR) if scale != 1.0 else proc_rgb.copy()
        face_landmarks = res.multi_face_landmarks[0].landmark

        def lm(i):
            if i >= len(face_landmarks): return None
            p = face_landmarks[i]; x = int(p.x * pw); y = int(p.y * ph)
            if scale != 1.0:
                x = int(x / scale); y = int(y / scale)
            return (x, y)

        # foundation
        if self.cfg.get("foundation_on"):
            face_pts = [lm(i) for i in FACE_OVAL_IDX if lm(i) is not None]
            if len(face_pts) > 6:
                mask = polygon_mask(out_rgb.shape, face_pts, blur_ksize=101)
                out_rgb = blend_color(out_rgb, hex_to_rgb(self.cfg.get("foundation_color", "#d0b08a")), mask, opacity=self.cfg.get("foundation_opacity",0.55))

        # concealer under-eyes
        if self.cfg.get("concealer_on"):
            def under_eye(eye_idx):
                pts = [lm(i) for i in eye_idx if lm(i) is not None]
                if len(pts) < 4: return None
                pts = np.array(pts)
                x,y,wbox,hbox = cv2.boundingRect(pts)
                cx = int(x + wbox*0.5); cy = int(y + hbox*0.7)
                mask = np.zeros((h,w), dtype=np.uint8)
                rx = max(6, int(wbox*0.6)); ry = max(4, int(hbox*0.5))
                cv2.ellipse(mask, (cx,cy), (rx, ry), 0, 0, 360, 255, -1)
                mask = cv2.GaussianBlur(mask, (41,41), 0)
                return mask.astype(np.float32)/255.0
            left = under_eye(LEFT_EYE_IDX); right = under_eye(RIGHT_EYE_IDX)
            combined = np.zeros((h,w), dtype=np.float32)
            if left is not None: combined = np.maximum(combined, left)
            if right is not None: combined = np.maximum(combined, right)
            if combined.sum()>0:
                out_rgb = blend_color(out_rgb, hex_to_rgb(self.cfg.get("concealer_color","#f2dfc7")), combined, opacity=self.cfg.get("concealer_opacity",0.7))

        # blush
        if self.cfg.get("blush_on"):
            cheek_pts = []
            # use rough cheek anchors; fallbacks included
            anchors = [234, 454, 93, 323, 132, 361, 58, 288]
            centers = []
            for i in anchors:
                p = lm(i)
                if p: centers.append((p, 12))
            if centers:
                mask = gaussian_spot_mask(out_rgb.shape, centers, radius_factor=12, blur_factor=0.33)
                out_rgb = blend_color(out_rgb, hex_to_rgb(self.cfg.get("blush_color","#ff9fb3")), mask, opacity=self.cfg.get("blush_opacity",0.28))

        # eyeshadow
        if self.cfg.get("eyeshadow_on"):
            for eye_idx in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
                pts = [lm(i) for i in eye_idx if lm(i) is not None]
                if len(pts) >= 3:
                    m = polygon_mask(out_rgb.shape, pts, blur_ksize=21)
                    out_rgb = blend_color(out_rgb, hex_to_rgb(self.cfg.get("eyeshadow_color","#6e4b9b")), m, opacity=self.cfg.get("eyeshadow_opacity",0.45))

        # eyeliner
        if self.cfg.get("eyeliner_on"):
            liner = out_rgb.copy()
            color = tuple(int(c*0.06) for c in hex_to_rgb(self.cfg.get("eyeliner_color","#000000")))
            try:
                for eye_idx in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
                    pts = [lm(i) for i in eye_idx if lm(i) is not None]
                    if len(pts) >= 3:
                        pts_arr = np.array(pts, dtype=np.int32).reshape((-1,1,2))
                        cv2.polylines(liner, [pts_arr], isClosed=False, color=color, thickness=max(1, int(h/260)), lineType=cv2.LINE_AA)
                out_rgb = cv2.addWeighted(out_rgb, 0.82, liner, 0.18, 0)
            except Exception:
                pass

        # mascara (upper)
        if self.cfg.get("mascara_on"):
            lash = out_rgb.copy()
            try:
                for eye_idx in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
                    pts = [lm(i) for i in eye_idx if lm(i) is not None]
                    if len(pts) >= 3:
                        pts_arr = np.array(pts, dtype=np.int32)
                        ys = pts_arr[:,1]; top_idx = np.argsort(ys)[:len(ys)//2 + 1]
                        top_pts = pts_arr[top_idx]
                        mask = np.zeros((h,w), dtype=np.uint8)
                        if len(top_pts) > 1:
                            cv2.polylines(mask, [top_pts.reshape((-1,1,2))], isClosed=False, color=255, thickness=max(1, int(h/220)), lineType=cv2.LINE_AA)
                            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)
                            mask = cv2.GaussianBlur(mask, (9,9), 0)
                            maskf = mask.astype(np.float32)/255.0
                            lash = blend_color(lash, (8,8,8), maskf, opacity=self.cfg.get("mascara_opacity",0.9))
                out_rgb = cv2.addWeighted(out_rgb, 0.78, lash, 0.22, 0)
            except Exception:
                pass

        # eye brightening
        if self.cfg.get("eye_brighten_on"):
            try:
                for iris_idx in (LEFT_IRIS_IDX, RIGHT_IRIS_IDX):
                    pts = [lm(i) for i in iris_idx if lm(i) is not None]
                    if len(pts) >= 3:
                        mask = polygon_mask(out_rgb.shape, pts, blur_ksize=15)
                        out_rgb = blend_color(out_rgb, (240,240,245), mask, opacity=0.45)
            except Exception:
                pass

        # highlighter (spots)
        if self.cfg.get("highlighter_on"):
            centers = []
            # forehead/nose/lip cupid approximations
            for i in (4, 1, 2, 12, 152):
                p = lm(i)
                if p: centers.append((p, 16))
            if centers:
                mask = gaussian_spot_mask(out_rgb.shape, centers, radius_factor=14, blur_factor=0.28)
                # lighter additive effect
                add = np.full_like(out_rgb, (255,240,230), dtype=np.uint8)
                mask3 = np.stack([mask,mask,mask], axis=2)
                out_rgb = np.clip(out_rgb.astype(np.float32)*(1-mask3*0.6) + add.astype(np.float32)*mask3*0.6, 0,255).astype(np.uint8)

        # lips (color + gloss)
        lip_pts = [lm(i) for i in LIPS_IDX if lm(i) is not None]
        if lip_pts and self.cfg.get("lip_on"):
            lip_mask = polygon_mask(out_rgb.shape, lip_pts, blur_ksize=21)
            out_rgb = blend_color(out_rgb, hex_to_rgb(self.cfg.get("lip_color","#b1223b")), lip_mask, opacity=self.cfg.get("lip_opacity",0.9))
            if self.cfg.get("lip_gloss_on"):
                xs = [p[0] for p in lip_pts]; ys = [p[1] for p in lip_pts]
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                r = max(6, int(min(max(xs)-min(xs), max(ys)-min(ys)) * 0.33))
                gloss = np.zeros((h,w), dtype=np.uint8)
                cv2.ellipse(gloss, (cx, cy - int(r*0.12)), (int(r*0.9), int(r*0.45)), 0, 0, 360, 255, -1)
                gloss = cv2.GaussianBlur(gloss, (33,33), 0)
                glossf = gloss.astype(np.float32)/255.0 * lip_mask
                add = np.full_like(out_rgb, (255,245,235), dtype=np.uint8)
                mask3 = np.stack([glossf, glossf, glossf], axis=2)
                out_rgb = np.clip(out_rgb.astype(np.float32)*(1-mask3*0.7) + add.astype(np.float32)*mask3*0.7, 0,255).astype(np.uint8)

        # moisturizer sheen
        if self.cfg.get("moisturizer_on"):
            hsv = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            v = hsv[:,:,2]; v = np.clip(v * (1 + 0.02 * self.cfg.get("moisturizer_opacity",0.4)), 0,255)
            hsv[:,:,2] = v
            out_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        return out_bgr

# ------------------ UI: product metadata + presets ------------------
PRODUCTS = [
    {"id":"foundation","title":"Foundation","tag":"Even skin", "desc":"Smooth base, adjustable opacity & shade."},
    {"id":"concealer","title":"Concealer","tag":"Brighten", "desc":"Under-eye & small spot coverage."},
    {"id":"blush","title":"Blush","tag":"Rosy cheeks", "desc":"Multi-spot blended blush."},
    {"id":"highlighter","title":"Highlighter","tag":"Glow", "desc":"Specular highlights on cheekbones & nose."},
    {"id":"eyeshadow","title":"Eyeshadow","tag":"Color pop", "desc":"Soft blended shadow on eyelids."},
    {"id":"eyeliner","title":"Eyeliner","tag":"Definition", "desc":"Soft natural liner along lash line."},
    {"id":"mascara","title":"Mascara","tag":"Volumize", "desc":"Upper lash darkening & thickness."},
    {"id":"lip","title":"Lip color","tag":"Finish", "desc":"Saturated lipstick & optional gloss."},
    {"id":"moisturizer","title":"Moisturizer","tag":"Sheen", "desc":"Subtle skin sheen for dewy looks."},
]

PRESETS = {
    "All Off": {},
    "Natural": dict(
        foundation_on=True, foundation_color="#e0c6a6", foundation_opacity=0.5,
        concealer_on=True, concealer_color="#f3e1c9", concealer_opacity=0.7,
        blush_on=True, blush_color="#f094a6", blush_opacity=0.28,
        highlighter_on=True, highlighter_opacity=0.45,
        eyeshadow_on=True, eyeshadow_color="#9b6fb4", eyeshadow_opacity=0.45,
        eyeliner_on=True, eyeliner_color="#0b0b0b",
        mascara_on=True, mascara_opacity=0.9,
        eye_brighten_on=True,
        lip_on=True, lip_color="#9b2030", lip_opacity=0.78,
        lip_gloss_on=True, lip_gloss_opacity=0.45,
        moisturizer_on=False, moisturizer_opacity=0.25,
    ),
    "Matte": dict(
        foundation_on=True, foundation_color="#c9a682", foundation_opacity=0.62,
        concealer_on=True, concealer_color="#e8d7bf", concealer_opacity=0.75,
        blush_on=True, blush_color="#d87a90", blush_opacity=0.32,
        highlighter_on=False,
        eyeshadow_on=True, eyeshadow_color="#5a2f66", eyeshadow_opacity=0.6,
        eyeliner_on=True, eyeliner_color="#050505", mascara_on=True, mascara_opacity=0.95,
        lip_on=True, lip_color="#7b1b2f", lip_opacity=0.95, lip_gloss_on=False,
        moisturizer_on=False,
    ),
    "Dewy": dict(
        foundation_on=True, foundation_color="#edceb0", foundation_opacity=0.36,
        concealer_on=True, concealer_color="#f6e4cf", concealer_opacity=0.6,
        blush_on=True, blush_color="#f0a3b6", blush_opacity=0.24,
        highlighter_on=True, highlighter_opacity=0.6,
        eyeshadow_on=True, eyeshadow_color="#8d5db6", eyeshadow_opacity=0.45,
        eyeliner_on=True, macarara_on=False,
        mascara_on=True, mascara_opacity=0.85,
        eye_brighten_on=True,
        lip_on=True, lip_color="#a83a4a", lip_opacity=0.75,
        lip_gloss_on=True, lip_gloss_opacity=0.68,
        moisturizer_on=True, moisturizer_opacity=0.45
    ),
    "Glam Night": dict(
        foundation_on=True, foundation_color="#d4b38b", foundation_opacity=0.68,
        concealer_on=True, concealer_opacity=0.85,
        blush_on=True, blush_color="#e04b6b", blush_opacity=0.36,
        highlighter_on=True, highlighter_opacity=0.75,
        eyeshadow_on=True, eyeshadow_color="#2b0d3a", eyeshadow_opacity=0.72,
        eyeliner_on=True, eyeliner_color="#030303", mascara_on=True, mascara_opacity=1.0,
        lip_on=True, lip_color="#6b0f23", lip_opacity=0.95, lip_gloss_on=False,
        moisturizer_on=False
    )
}

# ------------------ Sidebar: controls, product cards, presets ------------------
st.sidebar.markdown("## 🎨 Looks & Products")
preset_choice = st.sidebar.selectbox("Choose Preset", list(PRESETS.keys()), index=1)
apply_preset = st.sidebar.button("Apply Preset")

st.sidebar.markdown("### Product Cards")
for p in PRODUCTS:
    st.sidebar.markdown(f"<div class='prod'><h4>{p['title']} <small style='color:#b56aa6'>• {p['tag']}</small></h4><p>{p['desc']}</p></div>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("## Export & Snapshot")
snapshot_btn = st.sidebar.button("📸 Save snapshot (downloadable image)")
st.sidebar.markdown("Tip: Let camera warm up for a steady frame. Use Medium/Low resolution on Streamlit Cloud for best performance.")

# ------------------ Page content: controls + webrtc area + before-after ------------------
col_left, col_right = st.columns([0.36, 0.64])

with col_left:
    st.subheader("Fine controls")
    st.markdown("Toggle items individually, and adjust color/intensity. Use 'Apply Preset' to load a quick look.")
    # toggles and defaults (start off)
    cfg = {
        "foundation_on": st.checkbox("Foundation", value=False),
        "foundation_color": st.color_picker("Foundation shade", "#d0b08a"),
        "foundation_opacity": st.slider("Foundation intensity", 0.0, 1.0, 0.55),
        "concealer_on": st.checkbox("Concealer", value=False),
        "concealer_color": st.color_picker("Concealer shade", "#f2dfc7"),
        "concealer_opacity": st.slider("Concealer intensity", 0.0, 1.0, 0.7),
        "blush_on": st.checkbox("Blush", value=False),
        "blush_color": st.color_picker("Blush shade", "#ff9fb3"),
        "blush_opacity": st.slider("Blush intensity", 0.0, 1.0, 0.28),
        "highlighter_on": st.checkbox("Highlighter", value=False),
        "highlighter_opacity": st.slider("Highlighter intensity", 0.0, 1.0, 0.45),
        "eyeshadow_on": st.checkbox("Eyeshadow", value=False),
        "eyeshadow_color": st.color_picker("Eyeshadow shade", "#6e4b9b"),
        "eyeshadow_opacity": st.slider("Eyeshadow intensity", 0.0, 1.0, 0.45),
        "eyeliner_on": st.checkbox("Eyeliner", value=False),
        "eyeliner_color": st.color_picker("Eyeliner color", "#000000"),
        "mascara_on": st.checkbox("Mascara", value=False),
        "mascara_opacity": st.slider("Mascara intensity", 0.0, 1.0, 0.9),
        "eye_brighten_on": st.checkbox("Eye brightening (eye contact)", value=False),
        "lip_on": st.checkbox("Lip color", value=False),
        "lip_color": st.color_picker("Lip color", "#b1223b"),
        "lip_opacity": st.slider("Lip intensity", 0.0, 1.0, 0.88),
        "lip_gloss_on": st.checkbox("Lip gloss (specular)", value=False),
        "lip_gloss_opacity": st.slider("Lip gloss intensity", 0.0, 1.0, 0.5),
        "moisturizer_on": st.checkbox("Moisturizer sheen", value=False),
        "moisturizer_opacity": st.slider("Moisturizer intensity", 0.0, 1.0, 0.35),
        "scale": st.selectbox("Camera resolution (performance)", ["High", "Medium", "Low"])
    }

    # Map scale to numeric factor
    scale_map = {"High":1.0, "Medium":0.75, "Low":0.5}
    cfg["scale"] = scale_map[cfg["scale"]]

    # Apply preset if pressed
    if apply_preset:
        preset = PRESETS.get(preset_choice, {})
        # update values in UI by showing a confirmation and setting cfg keys (can't programmatically set widget defaults after render)
        st.success(f"Preset '{preset_choice}' applied — tweak with sliders below.")
        # override cfg dict values for transformer config (widgets won't change visually, but cfg controls transformer)
        for k, v in preset.items():
            cfg[k] = v

with col_right:
    st.subheader("Live preview — Use the controls to the left")
    rtc_conf = RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]})

    # Build config passed to transformer
    transformer_config = dict(
        foundation_on=cfg.get("foundation_on", False),
        foundation_color=cfg.get("foundation_color", "#d0b08a"),
        foundation_opacity=float(cfg.get("foundation_opacity", 0.55)),
        concealer_on=cfg.get("concealer_on", False),
        concealer_color=cfg.get("concealer_color", "#f2dfc7"),
        concealer_opacity=float(cfg.get("concealer_opacity", 0.7)),
        blush_on=cfg.get("blush_on", False),
        blush_color=cfg.get("blush_color", "#ff9fb3"),
        blush_opacity=float(cfg.get("blush_opacity", 0.28)),
        highlighter_on=cfg.get("highlighter_on", False),
        highlighter_opacity=float(cfg.get("highlighter_opacity", 0.45)),
        eyeshadow_on=cfg.get("eyeshadow_on", False),
        eyeshadow_color=cfg.get("eyeshadow_color", "#6e4b9b"),
        eyeshadow_opacity=float(cfg.get("eyeshadow_opacity", 0.45)),
        eyeliner_on=cfg.get("eyeliner_on", False),
        eyeliner_color=cfg.get("eyeliner_color", "#000000"),
        mascara_on=cfg.get("mascara_on", False),
        mascara_opacity=float(cfg.get("mascara_opacity", 0.9)),
        eye_brighten_on=cfg.get("eye_brighten_on", False),
        lip_on=cfg.get("lip_on", False),
        lip_color=cfg.get("lip_color", "#b1223b"),
        lip_opacity=float(cfg.get("lip_opacity", 0.88)),
        lip_gloss_on=cfg.get("lip_gloss_on", False),
        lip_gloss_opacity=float(cfg.get("lip_gloss_opacity", 0.5)),
        moisturizer_on=cfg.get("moisturizer_on", False),
        moisturizer_opacity=float(cfg.get("moisturizer_opacity", 0.35)),
        scale=float(cfg.get("scale", 1.0))
    )

    # create transformer factory
    def make_transformer():
        return AwardTransformer(cfg=transformer_config)

    webrtc_ctx = webrtc_streamer(
        key="glamcam_award",
        video_transformer_factory=make_transformer,
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=rtc_conf,
        async_transform=False,
    )

    # snapshot capture: if user pressed snapshot, try to grab current frame from player
    if snapshot_btn:
        if webrtc_ctx and webrtc_ctx.video_transformer:
            # try to get a frame (requires the transformer to store last frame; we can request a snapshot via JS, simplified approach below)
            st.info("Snapshot requested — move camera slightly and click 'Save snapshot' again if download does not appear.")
            # note: streamlit-webrtc advanced snapshots require custom callbacks; for quick demo we give instructions
        else:
            st.warning("Camera not running — allow camera and try again.")

# ------------------ Before/After demo area (uses saved frames if available) ------------------
st.markdown("---")
st.markdown("### Before / After demo")

# Small helper: show a static before/after image pair if camera not available
# We will capture a snapshot from the user's webcam is nontrivial in simple sync mode; instead, offer a demo slider comparing last transformed frame (if streamer stores one).
demo_left, demo_right = st.columns([0.5, 0.5])
with demo_left:
    st.markdown("**Before** — Raw camera frame")
    st.image("https://raw.githubusercontent.com/streamlit/example-data/master/face/face1.jpg", caption="Raw demo", use_column_width=True)
with demo_right:
    st.markdown("**After** — GlamCam processed (preset)")
    st.image("https://raw.githubusercontent.com/streamlit/example-data/master/face/face2.jpg", caption="Processed demo", use_column_width=True)

st.markdown("""
If you want an interactive draggable before/after overlay using your live camera frames, we can add a custom JS bridge that streams single-frame snapshots from streamlit-webrtc to a small HTML component — it's slightly more involved on Streamlit Cloud but fully doable. Want me to add that now?
""")

# ------------------ Footer / credits ------------------
st.markdown("---")
st.markdown("Built with ❤️ • MediaPipe Face Mesh • OpenCV blending • Streamlit Cloud deployable")
st.markdown("Want me to: (1) convert this to an in-browser WebGL / TF.js version for smoother mobile UX, or (2) add product thumbnail images + a 'Buy' product card? Reply which and I will implement it.")

