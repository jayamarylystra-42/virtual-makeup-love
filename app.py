"""
GlamCam — Dual-mode: Reliable Live + High-quality Photo Mode
- Live Mode: light, real-time makeup (lower-res, fast).
- Photo Mode: upload a selfie and apply studio-grade blending (seamlessClone + feathered masks).
Deploy notes: use Python 3.10, pinned requirements (see instructions).
"""
import streamlit as st
st.set_page_config(page_title="GlamCam — Live & Photo", layout="wide", page_icon="💄")

# ---- imports ----
try:
    import cv2
    import numpy as np
    import mediapipe as mp
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase, RTCConfiguration
    from PIL import Image
    import io, base64
except Exception as e:
    st.error(f"Import error: {e}\nMake sure requirements.txt contains pinned versions and runtime is Python 3.10.")
    st.stop()

# ---- helpers ----
def hex_to_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def polygon_mask(shape, polygon_pts, blur_ksize=41):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if not polygon_pts or len(polygon_pts) < 3:
        return mask.astype(float)
    cv2.fillPoly(mask, [np.array(polygon_pts, dtype=np.int32)], 255)
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    k = min(k, max(1, (min(shape[:2]) // 2) | 1))
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return (mask.astype(np.float32) / 255.0)

def gaussian_spot_mask(shape, centers, radius_factor=12, blur_factor=0.45):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for (cx, cy), rfac in centers:
        r = max(4, min(h, w) // rfac)
        cv2.circle(mask, (int(cx), int(cy)), int(r), 255, -1)
    k = int(min(h, w) * blur_factor)
    if k % 2 == 0: k += 1
    k = max(3, k)
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return (mask.astype(np.float32) / 255.0)

def blend_color(image_rgb, color_rgb, mask, opacity=0.6):
    overlay = np.full_like(image_rgb, color_rgb, dtype=np.uint8)
    mask3 = np.stack([mask, mask, mask], axis=2)
    blended = (image_rgb.astype(np.float32) * (1 - mask3 * opacity) + overlay.astype(np.float32) * (mask3 * opacity))
    return np.clip(blended, 0, 255).astype(np.uint8)

# ---- mediapipe indices ----
mp_face_mesh = mp.solutions.face_mesh
def idxs_from_connections(conns):
    s = set()
    for a,b in conns:
        s.add(a); s.add(b)
    return sorted(s)

LIPS_IDX = idxs_from_connections(mp_face_mesh.FACEMESH_LIPS)
LEFT_EYE_IDX = idxs_from_connections(mp_face_mesh.FACEMESH_LEFT_EYE)
RIGHT_EYE_IDX = idxs_from_connections(mp_face_mesh.FACEMESH_RIGHT_EYE)
FACE_OVAL_IDX = idxs_from_connections(mp_face_mesh.FACEMESH_FACE_OVAL)
LEFT_IRIS_IDX = idxs_from_connections(mp_face_mesh.FACEMESH_LEFT_IRIS)
RIGHT_IRIS_IDX = idxs_from_connections(mp_face_mesh.FACEMESH_RIGHT_IRIS)

# ---------------- Live Transformer (lightweight) ----------------
class LiveTransformer(VideoTransformerBase):
    def __init__(self, cfg):
        self.cfg = cfg
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                               refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    def transform(self, frame):
        try:
            img_bgr = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            scale = float(self.cfg.get("scale", 1.0))
            if scale != 1.0:
                small = cv2.resize(img_rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
                proc = small; ph, pw = small.shape[:2]
            else:
                proc = img_rgb; ph, pw = h, w

            res = self.face_mesh.process(proc)
            if not res.multi_face_landmarks:
                return img_bgr

            out = cv2.resize(proc, (w,h), interpolation=cv2.INTER_LINEAR) if scale != 1.0 else proc.copy()
            landmarks = res.multi_face_landmarks[0].landmark
            def lm(i):
                if i >= len(landmarks): return None
                p = landmarks[i]; x = int(p.x * pw); y = int(p.y * ph)
                if scale != 1.0:
                    x = int(x / scale); y = int(y / scale)
                return (x,y)

            # minimal but good blending: foundation (soft), blush spots, eyeliner/mascara simple
            # foundation
            if self.cfg.get("foundation_on"):
                face_pts = [lm(i) for i in FACE_OVAL_IDX if lm(i) is not None]
                if len(face_pts) > 6:
                    mask = polygon_mask(out.shape, face_pts, blur_ksize=61)
                    out = blend_color(out, hex_to_rgb(self.cfg.get("foundation_color","#e0c6a6")), mask, opacity=self.cfg.get("foundation_opacity",0.45))

            # blush
            if self.cfg.get("blush_on"):
                anchors = [234,454,93,323,132,361,58,288]
                centers=[]
                for i in anchors:
                    p = lm(i)
                    if p: centers.append((p, 14))
                if centers:
                    mask = gaussian_spot_mask(out.shape, centers, radius_factor=12, blur_factor=0.35)
                    out = blend_color(out, hex_to_rgb(self.cfg.get("blush_color","#f094a6")), mask, opacity=self.cfg.get("blush_opacity",0.25))

            # eyeliner & mascara (thin lines)
            if self.cfg.get("eyeliner_on") or self.cfg.get("mascara_on"):
                liner = out.copy()
                try:
                    for eye_idx in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
                        pts = [lm(i) for i in eye_idx if lm(i) is not None]
                        if len(pts) >= 3:
                            pts_arr = np.array(pts, dtype=np.int32).reshape((-1,1,2))
                            thickness = max(1, int(h/300))
                            color = (10,10,10)
                            cv2.polylines(liner, [pts_arr], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
                    out = cv2.addWeighted(out, 0.82, liner, 0.18, 0)
                except Exception:
                    pass

            # lips (simple color)
            lip_pts = [lm(i) for i in LIPS_IDX if lm(i) is not None]
            if lip_pts and self.cfg.get("lip_on"):
                lip_mask = polygon_mask(out.shape, lip_pts, blur_ksize=21)
                out = blend_color(out, hex_to_rgb(self.cfg.get("lip_color","#b1223b")), lip_mask, opacity=self.cfg.get("lip_opacity",0.85))
                # gloss: small specular ellipse
                if self.cfg.get("lip_gloss_on"):
                    xs = [p[0] for p in lip_pts]; ys=[p[1] for p in lip_pts]
                    cx, cy = int(np.mean(xs)), int(np.mean(ys))
                    r = max(6, int(min(max(xs)-min(xs), max(ys)-min(ys)) * 0.33))
                    gloss = np.zeros((h,w), dtype=np.uint8)
                    cv2.ellipse(gloss,(cx,cy - int(r*0.12)),(int(r*0.9),int(r*0.45)),0,0,360,255,-1)
                    gloss = cv2.GaussianBlur(gloss,(31,31),0)
                    glossf = gloss.astype(np.float32)/255.0 * lip_mask
                    add = np.full_like(out,(255,245,235),dtype=np.uint8)
                    m3 = np.stack([glossf,glossf,glossf],axis=2)
                    out = np.clip(out.astype(np.float32)*(1-m3*0.7) + add.astype(np.float32)*m3*0.7,0,255).astype(np.uint8)

            return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        except Exception:
            # if any runtime error, return original frame to avoid crash loop
            return frame.to_ndarray(format="bgr24")

# ---------------- Photo mode processing: higher-quality techniques ----------------
# Approach summary:
#  - Use Mediapipe to get precise landmarks on the full resolution image.
#  - For foundation and lips use cv2.seamlessClone (NORMAL_CLONE) with feathered masks so texture blends naturally.
#  - For blush/highlight/eyeshadow use gaussian spot masks + additive/specular highlight.
#  - This runs once on the uploaded photo and yields much better results than real-time approximations.

mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

def apply_makeup_photo(pil_img, cfg):
    """
    pil_img: Pillow Image (RGB)
    cfg: dict with makeup options and hex colors & opacities
    returns: PIL Image (RGB) with applied makeup (high-quality)
    """
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]

    # run mediapipe (full resolution)
    results = mp_face.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not results.multi_face_landmarks:
        return pil_img  # nothing detected

    lm = results.multi_face_landmarks[0].landmark
    def lmxy(i):
        if i >= len(lm): return None
        p = lm[i]
        return int(p.x * w), int(p.y * h)

    out = img.copy()

    # ---- Foundation (seamlessClone) ----
    if cfg.get("foundation_on"):
        face_pts = [lmxy(i) for i in FACE_OVAL_IDX if lmxy(i) is not None]
        if len(face_pts) > 6:
            # build colored layer same size
            color_rgb = hex_to_rgb(cfg.get("foundation_color", "#e0c6a6"))
            layer = np.full_like(out, color_rgb, dtype=np.uint8)
            # mask (feathered)
            mask = np.zeros((h,w), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(face_pts, dtype=np.int32)], 255)
            mask = cv2.GaussianBlur(mask, (101,101), 0)
            center = (int(np.mean([p[0] for p in face_pts])), int(np.mean([p[1] for p in face_pts])))
            # do seamless clone for better texture
            try:
                normal_mask = (mask>25).astype(np.uint8)*255
                cloned = cv2.seamlessClone(layer, out, normal_mask, center, cv2.NORMAL_CLONE)
                # blend cloned result with original by opacity
                alpha = cfg.get("foundation_opacity", 0.6)
                out = (out.astype(np.float32)*(1-alpha) + cloned.astype(np.float32)*alpha).astype(np.uint8)
            except Exception:
                # fallback to simple blend
                maskf = mask.astype(np.float32)/255.0
                out = blend_color(out, color_rgb, maskf, opacity=cfg.get("foundation_opacity",0.6))

    # ---- Lips (seamlessClone per lip mask) ----
    lip_points = [lmxy(i) for i in LIPS_IDX if lmxy(i) is not None]
    if lip_points and cfg.get("lip_on"):
        lip_mask = np.zeros((h,w), dtype=np.uint8)
        cv2.fillPoly(lip_mask, [np.array(lip_points, dtype=np.int32)], 255)
        lip_mask = cv2.GaussianBlur(lip_mask, (31,31), 0)
        color_rgb = hex_to_rgb(cfg.get("lip_color", "#b1223b"))
        layer = np.full_like(out, color_rgb, dtype=np.uint8)
        # center for clone near the lip center
        xs = [p[0] for p in lip_points]; ys = [p[1] for p in lip_points]
        center = (int(np.mean(xs)), int(np.mean(ys)))
        try:
            normal_mask = (lip_mask>20).astype(np.uint8)*255
            cloned = cv2.seamlessClone(layer, out, normal_mask, center, cv2.NORMAL_CLONE)
            alpha = cfg.get("lip_opacity", 0.9)
            out = (out.astype(np.float32)*(1-alpha) + cloned.astype(np.float32)*alpha).astype(np.uint8)
        except Exception:
            maskf = lip_mask.astype(np.float32)/255.0
            out = blend_color(out, color_rgb, maskf, opacity=cfg.get("lip_opacity",0.9))

        # gloss highlight
        if cfg.get("lip_gloss_on"):
            gloss_mask = np.zeros((h,w), dtype=np.uint8)
            rx = max(8, int((max(xs)-min(xs))*0.4))
            ry = max(4, int((max(ys)-min(ys))*0.2))
            cv2.ellipse(gloss_mask, (center[0], center[1]-int(ry*0.2)), (rx, ry), 0, 0, 360, 255, -1)
            gloss_mask = cv2.GaussianBlur(gloss_mask, (61,61), 0)
            glossf = gloss_mask.astype(np.float32)/255.0 * (lip_mask.astype(np.float32)/255.0)
            add = np.full_like(out, (255,245,235), dtype=np.uint8)
            m3 = np.stack([glossf, glossf, glossf], axis=2)
            out = np.clip(out.astype(np.float32)*(1-m3*0.8) + add.astype(np.float32)*m3*0.8, 0,255).astype(np.uint8)

    # ---- Blush (multi-spot high-quality) ----
    if cfg.get("blush_on"):
        anchors = [234, 454, 93, 323, 132, 361, 58, 288]
        centers = []
        for i in anchors:
            p = lmxy(i)
            if p: centers.append((p, 12))
        if centers:
            mask = gaussian_spot_mask(out.shape, centers, radius_factor=12, blur_factor=0.3)
            color_rgb = hex_to_rgb(cfg.get("blush_color","#f094a6"))
            # subtle multiply effect for realism
            overlay = (np.array(color_rgb, dtype=np.float32) / 255.0)
            out_float = out.astype(np.float32)/255.0
            out_float = out_float*(1 - mask[...,None]*0.18) + (overlay * (mask[...,None]*0.18))
            out = np.clip(out_float*255.0, 0,255).astype(np.uint8)

    # ---- Highlighter (specular spots) ----
    if cfg.get("highlighter_on"):
        centers = []
        for i in (4,1,2,12,152):
            p = lmxy(i)
            if p: centers.append((p, 16))
        if centers:
            mask = gaussian_spot_mask(out.shape, centers, radius_factor=12, blur_factor=0.25)
            add = np.full_like(out, (255,245,230), dtype=np.uint8)
            m3 = np.stack([mask,mask,mask], axis=2)
            out = np.clip(out.astype(np.float32)*(1-m3*0.65) + add.astype(np.float32)*m3*0.65, 0,255).astype(np.uint8)

    # ---- Eyeshadow (soft) ----
    if cfg.get("eyeshadow_on"):
        for eye_idx in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
            pts = [lmxy(i) for i in eye_idx if lmxy(i) is not None]
            if len(pts) >= 3:
                m = polygon_mask(out.shape, pts, blur_ksize=31)
                out = blend_color(out, hex_to_rgb(cfg.get("eyeshadow_color","#6e4b9b")), m, opacity=cfg.get("eyeshadow_opacity",0.5))

    return Image.fromarray(out)

# ---------------- UI ----------------
st.title("GlamCam — Live + Pro Photo Makeup")
st.write("Choose Live Mode (real-time) or Photo Mode (recommended for studio-quality results). Photo Mode will produce the best, realistic makeup.")

mode = st.radio("Mode", ["Live Mode (low-latency)", "Photo Mode (high-quality)"])

# Shared controls (subset)
col1, col2 = st.columns([0.5, 0.5])
with col1:
    foundation_on = st.checkbox("Foundation", value=True)
    foundation_color = st.color_picker("Foundation shade", "#e0c6a6")
    foundation_opacity = st.slider("Foundation intensity", 0.0, 1.0, 0.6)
    blush_on = st.checkbox("Blush", value=True)
    blush_color = st.color_picker("Blush shade", "#f094a6")
    blush_opacity = st.slider("Blush intensity", 0.0, 1.0, 0.28)
with col2:
    lip_on = st.checkbox("Lip color", value=True)
    lip_color = st.color_picker("Lip color", "#b1223b")
    lip_opacity = st.slider("Lip intensity", 0.0, 1.0, 0.9)
    lip_gloss_on = st.checkbox("Lip gloss", value=True)

# additional controls
eyeshadow_on = st.checkbox("Eyeshadow", value=True)
eyeshadow_color = st.color_picker("Eyeshadow shade", "#6e4b9b")
eyeshadow_opacity = st.slider("Eyeshadow intensity", 0.0, 1.0, 0.45)
eyeliner_on = st.checkbox("Eyeliner", value=True)
mascara_on = st.checkbox("Mascara", value=True)
concealer_on = st.checkbox("Concealer", value=True)
highlighter_on = st.checkbox("Highlighter", value=True)
eye_brighten_on = st.checkbox("Eye brighten", value=True)
moisturizer_on = st.checkbox("Moisturizer sheen", value=False)

# Build cfg used in both modes
cfg = dict(
    foundation_on=foundation_on, foundation_color=foundation_color, foundation_opacity=foundation_opacity,
    blush_on=blush_on, blush_color=blush_color, blush_opacity=blush_opacity,
    lip_on=lip_on, lip_color=lip_color, lip_opacity=lip_opacity, lip_gloss_on=lip_gloss_on,
    eyeshadow_on=eyeshadow_on, eyeshadow_color=eyeshadow_color, eyeshadow_opacity=eyeshadow_opacity,
    eyeliner_on=eyeliner_on, mascara_on=mascara_on, concealer_on=concealer_on, highlighter_on=highlighter_on,
    eye_brighten_on=eye_brighten_on, moisturizer_on=moisturizer_on
)

# ---------- Live Mode ----------
if mode == "Live Mode (low-latency)":
    st.info("Live Mode is lower-latency and best for quick demos. For studio-quality looks use Photo Mode.")
    # performance/resolution
    res = st.selectbox("Webcam resolution", ["640x480", "320x240"], index=0)
    scale = 0.75 if res=="640x480" else 0.5
    cfg["scale"] = scale

    # RTC config: STUN + optional TURN
    ice = [{"urls":["stun:stun.l.google.com:19302"]}]
    TURN_URL = st.secrets.get("TURN_URL") if hasattr(st, "secrets") else None
    TURN_USER = st.secrets.get("TURN_USER") if hasattr(st, "secrets") else None
    TURN_PASS = st.secrets.get("TURN_PASS") if hasattr(st, "secrets") else None
    if TURN_URL and TURN_USER and TURN_PASS:
        ice.append({"urls":[TURN_URL], "username": TURN_USER, "credential": TURN_PASS})
        st.info("Using TURN from secrets (improves connectivity on strict networks).")

    rtc_conf = RTCConfiguration({"iceServers": ice})

    # media constraints (lower fps)
    if res == "640x480":
        w,h = 640,480
    else:
        w,h = 320,240
    media_constraints = {"video": {"width": w, "height": h, "frameRate": 12}, "audio": False}

    # start webrtc
    webrtc_ctx = webrtc_streamer(
        key="glam_live",
        video_transformer_factory=lambda: LiveTransformer(cfg),
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_conf,
        media_stream_constraints=media_constraints,
        async_transform=False,
        desired_playing_state=True,
    )

    if webrtc_ctx:
        state = webrtc_ctx.state
        st.write(f"WebRTC state: **{state}**")
        if state == "PLAYING":
            st.success("Camera streaming. If camera turns off, try lowering resolution or use Photo Mode.")
        elif state == "FAILED":
            st.warning("Player failed — try Photo Mode or add TURN credentials in Secrets.")
        else:
            st.info("Initializing... allow camera permission in the browser.")

# ---------- Photo Mode (high-quality) ----------
else:
    st.success("Photo Mode: upload a clear frontal selfie for the best results. This mode uses higher-quality blending.")
    uploaded = st.file_uploader("Upload a selfie (jpg/png) or drag here", type=["jpg","jpeg","png"])
    use_camera_snap = st.button("Capture snapshot from webcam (if camera working)")
    # If user captures snapshot from webcam, we try to get a frame via webrtc snapshot API (best-effort)
    # For simplicity: if webcam is available from a previously started webrtc_ctx, request snapshot. This is platform-limited.
    snapshot_img = None
    if use_camera_snap:
        st.info("If you recently allowed camera via Live Mode, your camera may be used to capture a snapshot. If not, upload an image.")
        # Try to capture from webrtc_ctx if present
        try:
            if "webrtc_ctx" in globals() and webrtc_ctx and webrtc_ctx.video_transformer:
                # try to use media state snapshot — not guaranteed on every environment
                pass
        except Exception:
            pass

    img_to_process = None
    if uploaded:
        img_to_process = Image.open(uploaded).convert("RGB")
    elif snapshot_img is not None:
        img_to_process = snapshot_img

    if img_to_process is not None:
        st.image(img_to_process, caption="Input (will be processed at full resolution)", use_column_width=False, width=360)
        if st.button("Apply makeup (High-quality)"):
            with st.spinner("Processing photo with high-quality blending..."):
                result_pil = apply_makeup_photo(img_to_process, cfg)
                st.image(result_pil, caption="Result — High-quality makeup", use_column_width=False, width=360)
                # download button
                buf = io.BytesIO()
                result_pil.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                href = f'<a href="data:file/png;base64,{b64}" download="glamcam_result.png">📥 Download result</a>'
                st.markdown(href, unsafe_allow_html=True)
    else:
        st.info("Upload a clear frontal selfie (good lighting) and click 'Apply makeup' for best results. Photo Mode gives the most realistic outcome.")

# ---- footer tips ----
st.markdown("---")
st.markdown("Tips: For best photo results use a high-resolution frontal selfie, good natural lighting, and remove glasses. If Live Mode camera keeps disconnecting, use Photo Mode — it's faster to produce pro-level results.")



