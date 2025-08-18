"""
GlamCam — Award UI (TURN-ready, lower-res by default, user-friendly diagnostics)

Replace your existing app.py with this file. For best reliability on Streamlit Cloud:
  1) Add an optional TURN server in Streamlit secrets (see below).
  2) Use Python 3.10 runtime and the pinned requirements in requirements.txt.

This file keeps the advanced makeup transformer and UI from earlier edits,
but improves WebRTC reliability and gives helpful troubleshooting hints.
"""

import streamlit as st
st.set_page_config(page_title="GlamCam — TURN-ready", layout="wide", page_icon="💄")

# ---------- Imports ----------
try:
    import cv2
    import numpy as np
    import mediapipe as mp
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase, RTCConfiguration
    from PIL import Image
    import base64, io
except Exception as e:
    st.error(
        "Import error: {}.\n\n"
        "Make sure your repo contains a working requirements.txt and that "
        "the Streamlit Cloud app uses Python 3.10.".format(e)
    )
    st.stop()

# ---------- Small helper functions used by transformer ----------
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

# ---------- Mediapipe indices ----------
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

# ---------- Simple transformer (keeps existing makeup features) ----------
class ReliableTransformer(VideoTransformerBase):
    def __init__(self, cfg):
        self.cfg = cfg
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                               max_num_faces=1,
                                               refine_landmarks=True,
                                               min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5)
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        scale = float(self.cfg.get("scale", 1.0))
        if scale != 1.0:
            small = cv2.resize(img_rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
            proc = small; ph, pw = small.shape[:2]
        else:
            proc = img_rgb; ph, pw = h, w

        results = self.face_mesh.process(proc)
        if not results.multi_face_landmarks:
            return img  # no face found, return original bgr

        out = cv2.resize(proc, (w, h), interpolation=cv2.INTER_LINEAR) if scale != 1.0 else proc.copy()
        lm_list = results.multi_face_landmarks[0].landmark

        def lm(i):
            if i >= len(lm_list): return None
            p = lm_list[i]
            x = int(p.x * pw); y = int(p.y * ph)
            if scale != 1.0:
                x = int(x / scale); y = int(y / scale)
            return (x, y)

        # foundation
        if self.cfg.get("foundation_on"):
            face_pts = [lm(i) for i in FACE_OVAL_IDX if lm(i) is not None]
            if len(face_pts) > 6:
                mask = polygon_mask(out.shape, face_pts, blur_ksize=101)
                out = blend_color(out, hex_to_rgb(self.cfg.get("foundation_color", "#d0b08a")), mask, opacity=self.cfg.get("foundation_opacity",0.55))

        # concealer
        if self.cfg.get("concealer_on"):
            def under_eye(eye_idx):
                pts = [lm(i) for i in eye_idx if lm(i) is not None]
                if len(pts) < 4: return None
                ptsa = np.array(pts)
                x,y,wbox,hbox = cv2.boundingRect(ptsa)
                cx = int(x + wbox*0.5); cy = int(y + hbox*0.7)
                mask = np.zeros((h,w), dtype=np.uint8)
                rx = max(6, int(wbox*0.6)); ry = max(4, int(hbox*0.5))
                cv2.ellipse(mask, (cx,cy), (rx, ry), 0,0,360,255,-1)
                mask = cv2.GaussianBlur(mask, (41,41), 0)
                return mask.astype(np.float32)/255.0
            left = under_eye(LEFT_EYE_IDX); right = under_eye(RIGHT_EYE_IDX)
            comb = np.zeros((h,w), dtype=np.float32)
            if left is not None: comb = np.maximum(comb, left)
            if right is not None: comb = np.maximum(comb, right)
            if comb.sum() > 0:
                out = blend_color(out, hex_to_rgb(self.cfg.get("concealer_color", "#f2dfc7")), comb, opacity=self.cfg.get("concealer_opacity",0.7))

        # blush (multi-spot)
        if self.cfg.get("blush_on"):
            anchors = [234, 454, 93, 323, 132, 361, 58, 288]
            centers = []
            for i in anchors:
                p = lm(i)
                if p: centers.append((p, 12))
            if centers:
                mask = gaussian_spot_mask(out.shape, centers, radius_factor=12, blur_factor=0.33)
                out = blend_color(out, hex_to_rgb(self.cfg.get("blush_color", "#ff9fb3")), mask, opacity=self.cfg.get("blush_opacity",0.28))

        # eyeshadow
        if self.cfg.get("eyeshadow_on"):
            for eye_idx in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
                pts = [lm(i) for i in eye_idx if lm(i) is not None]
                if len(pts) >= 3:
                    m = polygon_mask(out.shape, pts, blur_ksize=21)
                    out = blend_color(out, hex_to_rgb(self.cfg.get("eyeshadow_color", "#6e4b9b")), m, opacity=self.cfg.get("eyeshadow_opacity", 0.45))

        # eyeliner
        if self.cfg.get("eyeliner_on"):
            liner = out.copy()
            color = tuple(int(c*0.06) for c in hex_to_rgb(self.cfg.get("eyeliner_color", "#000000")))
            try:
                for eye_idx in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
                    pts = [lm(i) for i in eye_idx if lm(i) is not None]
                    if len(pts) >= 3:
                        pts_arr = np.array(pts, dtype=np.int32).reshape((-1,1,2))
                        cv2.polylines(liner, [pts_arr], isClosed=False, color=color, thickness=max(1, int(h/260)), lineType=cv2.LINE_AA)
                out = cv2.addWeighted(out, 0.82, liner, 0.18, 0)
            except Exception:
                pass

        # mascara (upper lash)
        if self.cfg.get("mascara_on"):
            lash = out.copy()
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
                out = cv2.addWeighted(out, 0.78, lash, 0.22, 0)
            except Exception:
                pass

        # eye brighten
        if self.cfg.get("eye_brighten_on"):
            try:
                for iris_idx in (LEFT_IRIS_IDX, RIGHT_IRIS_IDX):
                    pts = [lm(i) for i in iris_idx if lm(i) is not None]
                    if len(pts) >= 3:
                        m = polygon_mask(out.shape, pts, blur_ksize=15)
                        out = blend_color(out, (240,240,245), m, opacity=0.45)
            except Exception:
                pass

        # highlighter
        if self.cfg.get("highlighter_on"):
            centers = []
            for i in (4,1,2,12,152):
                p = lm(i)
                if p: centers.append((p, 16))
            if centers:
                mask = gaussian_spot_mask(out.shape, centers, radius_factor=14, blur_factor=0.28)
                add = np.full_like(out, (255,240,230), dtype=np.uint8)
                mask3 = np.stack([mask,mask,mask], axis=2)
                out = np.clip(out.astype(np.float32)*(1-mask3*0.6) + add.astype(np.float32)*mask3*0.6, 0,255).astype(np.uint8)

        # lips
        lip_pts = [lm(i) for i in LIPS_IDX if lm(i) is not None]
        if lip_pts and self.cfg.get("lip_on"):
            lip_mask = polygon_mask(out.shape, lip_pts, blur_ksize=21)
            out = blend_color(out, hex_to_rgb(self.cfg.get("lip_color", "#b1223b")), lip_mask, opacity=self.cfg.get("lip_opacity",0.9))
            if self.cfg.get("lip_gloss_on"):
                xs = [p[0] for p in lip_pts]; ys = [p[1] for p in lip_pts]
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                r = max(6, int(min(max(xs)-min(xs), max(ys)-min(ys)) * 0.33))
                gloss = np.zeros((h,w), dtype=np.uint8)
                cv2.ellipse(gloss, (cx, cy - int(r*0.12)), (int(r*0.9), int(r*0.45)), 0,0,360,255,-1)
                gloss = cv2.GaussianBlur(gloss, (33,33), 0)
                glossf = gloss.astype(np.float32)/255.0 * lip_mask
                add = np.full_like(out, (255,245,235), dtype=np.uint8)
                mask3 = np.stack([glossf,glossf,glossf], axis=2)
                out = np.clip(out.astype(np.float32)*(1-mask3*0.7) + add.astype(np.float32)*mask3*0.7, 0,255).astype(np.uint8)

        # moisturizer sheen
        if self.cfg.get("moisturizer_on"):
            hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
            v = hsv[:,:,2]; v = np.clip(v * (1 + 0.02 * self.cfg.get("moisturizer_opacity",0.4)), 0,255)
            hsv[:,:,2] = v
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

# ---------- Page UI ----------
st.title("✨ GlamCam — Reliable Live Makeup")
st.write("If webcam keeps loading: try Chrome, allow camera access, or add TURN credentials in app secrets (instructions below).")

# left controls (minimal for reliability demo; keep full controls in your app)
with st.sidebar:
    st.header("Performance & Network")
    st.markdown("Set lower resolution if your camera doesn't start.")
    resolution = st.selectbox("Camera resolution", ["640x480 (recommended)", "1280x720", "320x240"])
    scale_map = {"1280x720": 1.0, "640x480": 0.75, "320x240": 0.5}
    scale = scale_map[resolution]
    st.markdown("---")
    st.markdown("TURN server (optional, recommended if you are behind strict firewall).")
    st.markdown("Add these keys to Streamlit Secrets (Settings → Secrets) as `TURN_URL`, `TURN_USER`, `TURN_PASS`.")
    st.markdown("If you do not have TURN credentials, the player will try to use STUN only (works on most networks).")

# Build RTC configuration with optional TURN from secrets
ice_servers = [{"urls":["stun:stun.l.google.com:19302"]}]  # always include at least STUN

turn_url = st.secrets.get("TURN_URL") if hasattr(st, "secrets") else None
turn_user = st.secrets.get("TURN_USER") if hasattr(st, "secrets") else None
turn_pass = st.secrets.get("TURN_PASS") if hasattr(st, "secrets") else None

if turn_url and turn_user and turn_pass:
    try:
        ice_servers.append({"urls":[turn_url], "username": turn_user, "credential": turn_pass})
        st.sidebar.success("Using TURN from secrets")
    except Exception:
        st.sidebar.error("Could not add TURN from secrets — check secret keys")
else:
    st.sidebar.info("No TURN configured. Using STUN only (works on most networks).")

rtc_configuration = RTCConfiguration({"iceServers": ice_servers})

# Build transformer config (you can expand to map your full UI settings)
transformer_cfg = {
    "scale": scale,
    # default enable some features to demo
    "foundation_on": True,
    "foundation_color": "#e0c6a6",
    "foundation_opacity": 0.5,
    "concealer_on": True,
    "concealer_color": "#f3e1c9",
    "concealer_opacity": 0.6,
    "blush_on": True,
    "blush_color": "#f094a6",
    "blush_opacity": 0.25,
    "eyeshadow_on": True,
    "eyeshadow_color": "#6e4b9b",
    "eyeshadow_opacity": 0.45,
    "eyeliner_on": True,
    "eyeliner_color": "#000000",
    "mascara_on": True,
    "mascara_opacity": 0.9,
    "eye_brighten_on": True,
    "lip_on": True,
    "lip_color": "#b1223b",
    "lip_opacity": 0.85,
    "lip_gloss_on": True,
    "lip_gloss_opacity": 0.45,
    "highlighter_on": True,
    "highlighter_opacity": 0.45,
    "moisturizer_on": False,
    "moisturizer_opacity": 0.35
}

# Media constraints (reduce to avoid heavy CPU and connection issues)
media_constraints = {
    "video": {
        "width": 640 if resolution == "640x480 (recommended)" else (1280 if resolution=="1280x720" else 320),
        "height": 480 if resolution == "640x480 (recommended)" else (720 if resolution=="1280x720" else 240),
        "frameRate": 15
    },
    "audio": False
}

st.write("Stream status & troubleshooting tips appear below the player. If the player never starts, try switching resolution or adding TURN credentials in Secrets.")

# Launch webrtc player
webrtc_ctx = webrtc_streamer(
    key="glamcam_reliable",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    media_stream_constraints=media_constraints,
    video_transformer_factory=lambda: ReliableTransformer(transformer_cfg),
    async_transform=False,
    desired_playing_state=True,
)

# Provide runtime diagnostics to user
if webrtc_ctx:
    status = webrtc_ctx.state
    st.write(f"WebRTC state: **{status}**")
    if status == "PLAYING":
        st.success("Camera streaming. If you cannot see your face: make sure the correct camera is selected, and no other app is using it.")
    elif status in ("NOT_PLAYING", "FAILED"):
        st.warning("Player not running. Try selecting a lower resolution, refresh the page, or allow camera permission in the browser prompt.")
    else:
        st.info("Initializing camera... please allow camera access and wait a few seconds.")

    # helpful troubleshooting hints
    st.markdown("""
    **If camera does not start**
    1. Make sure you **clicked Allow** in the browser permission dialog (often a camera icon in the address bar).
    2. Try a different browser (Chrome or Edge recommended).
    3. If on a corporate / school network, add TURN credentials in Streamlit Secrets:
       - `TURN_URL` (e.g. `turn:your-turn-host:3478`)
       - `TURN_USER`
       - `TURN_PASS`
    4. If still failing, run locally with `streamlit run app.py` to confirm your camera works locally.
    """)

# Simple "how to add secrets" help (visible to user)
st.markdown("---")
st.subheader("Add TURN server (optional, improves reliability behind strict firewalls)")
st.markdown(
    "1. In Streamlit Cloud, open your app → **Settings → Secrets**.  \n"
    "2. Add keys:\n\n"
    "```\n"
    "TURN_URL = \"turn:your.turn.server:3478\"\n"
    "TURN_USER = \"USERNAME\"\n"
    "TURN_PASS = \"PASSWORD\"\n"
    "```\n"
    "3. Save and **re-deploy** the app (or click 'Restart')."
)
