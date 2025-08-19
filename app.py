"""
GlamCam — Award-style Streamlit live virtual makeup (full working app.py)
Instructions:
  - Add recommended requirements.txt (see comment below).
  - Optionally add TURN credentials in Streamlit Secrets: TURN_URL, TURN_USER, TURN_PASS
  - Deploy to Streamlit Cloud (use Python 3.10), allow camera in browser (Chrome recommended).
"""

import streamlit as st
st.set_page_config(page_title="GlamCam — Award UI", page_icon="💄", layout="wide", initial_sidebar_state="expanded")

# ---- imports ----
try:
    import cv2
    import numpy as np
    import mediapipe as mp
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase, RTCConfiguration
    from PIL import Image
    import io, base64
except Exception as e:
    st.error(f"Import error: {e}\n\nMake sure requirements.txt contains pinned versions and the Cloud runtime is Python 3.10.")
    st.stop()

# ---- small helpers ----
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

# ---- mediapipe indices ----
mp_face_mesh = mp.solutions.face_mesh
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

# ---- Video transformer that applies makeup to each frame ----
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
        scale = float(self.cfg.get("scale", 1.0))
        if scale != 1.0:
            small = cv2.resize(img_rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
            proc_rgb = small; ph, pw = small.shape[:2]
        else:
            proc_rgb = img_rgb; ph, pw = h, w

        res = self.face_mesh.process(proc_rgb)
        if not res.multi_face_landmarks:
            return img_bgr

        out_rgb = cv2.resize(proc_rgb, (w, h), interpolation=cv2.INTER_LINEAR) if scale != 1.0 else proc_rgb.copy()
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
                out_rgb = blend_color(out_rgb, hex_to_rgb(self.cfg.get("foundation_color", "#d0b08a")), mask, opacity=self.cfg.get("foundation_opacity", 0.55))

        # concealer (under-eye)
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
            if combined.sum() > 0:
                out_rgb = blend_color(out_rgb, hex_to_rgb(self.cfg.get("concealer_color", "#f2dfc7")), combined, opacity=self.cfg.get("concealer_opacity", 0.7))

        # blush
        if self.cfg.get("blush_on"):
            anchors = [234, 454, 93, 323, 132, 361, 58, 288]
            centers = []
            for i in anchors:
                p = lm(i)
                if p: centers.append((p, 12))
            if centers:
                mask = gaussian_spot_mask(out_rgb.shape, centers, radius_factor=12, blur_factor=0.33)
                out_rgb = blend_color(out_rgb, hex_to_rgb(self.cfg.get("blush_color", "#ff9fb3")), mask, opacity=self.cfg.get("blush_opacity", 0.28))

        # eyeshadow
        if self.cfg.get("eyeshadow_on"):
            for eye_idx in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
                pts = [lm(i) for i in eye_idx if lm(i) is not None]
                if len(pts) >= 3:
                    m = polygon_mask(out_rgb.shape, pts, blur_ksize=21)
                    out_rgb = blend_color(out_rgb, hex_to_rgb(self.cfg.get("eyeshadow_color", "#6e4b9b")), m, opacity=self.cfg.get("eyeshadow_opacity", 0.45))

        # eyeliner
        if self.cfg.get("eyeliner_on"):
            liner = out_rgb.copy()
            color = tuple(int(c * 0.06) for c in hex_to_rgb(self.cfg.get("eyeliner_color", "#000000")))
            try:
                for eye_idx in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
                    pts = [lm(i) for i in eye_idx if lm(i) is not None]
                    if len(pts) >= 3:
                        pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(liner, [pts_arr], isClosed=False, color=color, thickness=max(1, int(h/260)), lineType=cv2.LINE_AA)
                out_rgb = cv2.addWeighted(out_rgb, 0.82, liner, 0.18, 0)
            except Exception:
                pass

        # mascara
        if self.cfg.get("mascara_on"):
            lash = out_rgb.copy()
            try:
                for eye_idx in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
                    pts = [lm(i) for i in eye_idx if lm(i) is not None]
                    if len(pts) >= 3:
                        pts_arr = np.array(pts, dtype=np.int32)
                        ys = pts_arr[:,1]; top_idx = np.argsort(ys)[:len(ys)//2 + 1]
                        top_pts = pts_arr[top_idx]
                        mask = np.zeros((h, w), dtype=np.uint8)
                        if len(top_pts) > 1:
                            cv2.polylines(mask, [top_pts.reshape((-1,1,2))], isClosed=False, color=255, thickness=max(1, int(h/220)), lineType=cv2.LINE_AA)
                            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)
                            mask = cv2.GaussianBlur(mask, (9,9), 0)
                            maskf = mask.astype(np.float32)/255.0
                            lash = blend_color(lash, (8,8,8), maskf, opacity=self.cfg.get("mascara_opacity", 0.9))
                out_rgb = cv2.addWeighted(out_rgb, 0.78, lash, 0.22, 0)
            except Exception:
                pass

        # eye brighten
        if self.cfg.get("eye_brighten_on"):
            try:
                for iris_idx in (LEFT_IRIS_IDX, RIGHT_IRIS_IDX):
                    pts = [lm(i) for i in iris_idx if lm(i) is not None]
                    if len(pts) >= 3:
                        mask = polygon_mask(out_rgb.shape, pts, blur_ksize=15)
                        out_rgb = blend_color(out_rgb, (240,240,245), mask, opacity=0.45)
            except Exception:
                pass

        # highlighter
        if self.cfg.get("highlighter_on"):
            centers = []
            for i in (4, 1, 2, 12, 152):
                p = lm(i)
                if p: centers.append((p, 16))
            if centers:
                mask = gaussian_spot_mask(out_rgb.shape, centers, radius_factor=14, blur_factor=0.28)
                add = np.full_like(out_rgb, (255,240,230), dtype=np.uint8)
                mask3 = np.stack([mask,mask,mask], axis=2)
                out_rgb = np.clip(out_rgb.astype(np.float32)*(1-mask3*0.6) + add.astype(np.float32)*mask3*0.6, 0,255).astype(np.uint8)

        # lips + gloss
        lip_pts = [lm(i) for i in LIPS_IDX if lm(i) is not None]
        if lip_pts and self.cfg.get("lip_on"):
            lip_mask = polygon_mask(out_rgb.shape, lip_pts, blur_ksize=21)
            out_rgb = blend_color(out_rgb, hex_to_rgb(self.cfg.get("lip_color", "#b1223b")), lip_mask, opacity=self.cfg.get("lip_opacity", 0.9))
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
            v = hsv[:,:,2]; v = np.clip(v * (1 + 0.02 * self.cfg.get("moisturizer_opacity", 0.4)), 0,255)
            hsv[:,:,2] = v
            out_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

# ---- UI & presets (kept minimal here; your full UI can be added back) ----
st.title("✨ GlamCam — Live Virtual Makeup")
st.markdown("Use the left controls to toggle items. Allow camera permission when asked (Chrome recommended).")

# Simple left controls
col_l, col_r = st.columns([0.36, 0.64])

with col_l:
    st.header("Controls")
    foundation_on = st.checkbox("Foundation", value=True)
    foundation_color = st.color_picker("Foundation shade", "#e0c6a6")
    foundation_opacity = st.slider("Foundation intensity", 0.0, 1.0, 0.5)
    blush_on = st.checkbox("Blush", value=True)
    blush_color = st.color_picker("Blush shade", "#f094a6")
    blush_opacity = st.slider("Blush intensity", 0.0, 1.0, 0.28)
    eyeshadow_on = st.checkbox("Eyeshadow", value=True)
    eyeshadow_color = st.color_picker("Eyeshadow shade", "#6e4b9b")
    eyeliner_on = st.checkbox("Eyeliner", value=True)
    mascara_on = st.checkbox("Mascara", value=True)
    lip_on = st.checkbox("Lip color", value=True)
    lip_color = st.color_picker("Lip color", "#b1223b")
    lip_gloss_on = st.checkbox("Lip gloss", value=True)
    concealer_on = st.checkbox("Concealer", value=True)
    highlighter_on = st.checkbox("Highlighter", value=True)
    eye_brighten_on = st.checkbox("Eye brighten", value=True)
    moisturizer_on = st.checkbox("Moisturizer sheen", value=False)
    resolution = st.selectbox("Resolution", ["640x480", "1280x720", "320x240"], index=0)

with col_r:
    st.header("Live Preview")
    # build RTC config (use STUN + optional TURN from secrets)
    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    TURN_URL = st.secrets.get("TURN_URL") if hasattr(st, "secrets") else None
    TURN_USER = st.secrets.get("TURN_USER") if hasattr(st, "secrets") else None
    TURN_PASS = st.secrets.get("TURN_PASS") if hasattr(st, "secrets") else None
    if TURN_URL and TURN_USER and TURN_PASS:
        ice_servers.append({"urls":[TURN_URL], "username": TURN_USER, "credential": TURN_PASS})
        st.info("Using TURN server from secrets (improves connectivity behind strict NATs).")
    rtc_conf = RTCConfiguration({"iceServers": ice_servers})

    # map resolution
    if resolution == "1280x720":
        w, h = 1280, 720; scale = 1.0
    elif resolution == "320x240":
        w, h = 320, 240; scale = 0.5
    else:
        w, h = 640, 480; scale = 0.75

    # assemble transformer cfg from controls
    transformer_config = dict(
        foundation_on=foundation_on,
        foundation_color=foundation_color,
        foundation_opacity=float(foundation_opacity),
        blush_on=blush_on,
        blush_color=blush_color,
        blush_opacity=float(blush_opacity),
        eyeshadow_on=eyeshadow_on,
        eyeshadow_color=eyeshadow_color,
        eyeliner_on=eyeliner_on,
        mascara_on=mascara_on,
        lip_on=lip_on,
        lip_color=lip_color,
        lip_gloss_on=lip_gloss_on,
        concealer_on=concealer_on,
        highlighter_on=highlighter_on,
        eye_brighten_on=eye_brighten_on,
        moisturizer_on=moisturizer_on,
        moisturizer_opacity=0.35,
        scale=scale
    )

    # factory and start webrtc streamer
    def factory():
        return AwardTransformer(cfg=transformer_config)

    media_constraints = {
        "video": {"width": w, "height": h, "frameRate": 15},
        "audio": False
    }

    webrtc_ctx = webrtc_streamer(
        key="glamcam_live",
        video_transformer_factory=factory,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_conf,
        media_stream_constraints=media_constraints,
        async_transform=False,
    )

    # status / diagnostics
    if webrtc_ctx and webrtc_ctx.state:
        st.write(f"WebRTC state: **{webrtc_ctx.state}**")
        if webrtc_ctx.state == "PLAYING":
            st.success("Camera streaming — makeup overlays active.")
        else:
            st.info("Initializing camera... allow permission if prompted, or try another browser.")

# footer
st.markdown("---")
st.markdown("Built with MediaPipe Face Mesh + OpenCV blending. Tip: if the camera never starts, add TURN credentials in Streamlit Secrets and restart the app.")

# ---- requirements.txt recommended (copy to your repo as requirements.txt) ----
# streamlit>=1.20
# opencv-python-headless==4.7.0.72
# mediapipe==0.10.11
# numpy==1.26.4
# Pillow>=9.0
# streamlit-webrtc==0.53.0
# av==10.0.0

    
  
