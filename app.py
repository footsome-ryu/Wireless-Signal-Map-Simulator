import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# --- 1. Physics Model ---
FREQUENCY_MHZ = 447

def calculate_fspl(distance_m):
    if distance_m <= 0.5: return 0
    return 20 * np.log10(distance_m) + 20 * np.log10(FREQUENCY_MHZ) - 27.55

def count_walls_fast(start_pt, end_pt, wall_mask, step=5): # step을 줄여 벽 인식 정밀도 향상
    x0, y0 = int(start_pt[0]), int(start_pt[1])
    x1, y1 = int(end_pt[0]), int(end_pt[1])
    dist_px = np.sqrt((x1-x0)**2 + (y1-y0)**2)
    if dist_px < step: return 0
    num_samples = int(dist_px / step)
    xs = np.linspace(x0, x1, num_samples).astype(int)
    ys = np.linspace(y0, y1, num_samples).astype(int)
    h, w = wall_mask.shape
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    walls = np.sum(wall_mask[ys[valid], xs[valid]] > 0)
    return walls / 2.0

# --- 2. Setup ---
st.set_page_config(layout="wide", page_title="Wireless Signal Map Simulator")
st.title("📡 Wireless Signal Map Simulator")

if 'devices' not in st.session_state: st.session_state.devices = []
if 'last_click' not in st.session_state: st.session_state.last_click = None

# Sidebar
st.sidebar.header("Settings")
tx_eff = st.sidebar.slider("TX Power (dBm)", -10, 20, 0)
rp_eff = st.sidebar.slider("RP Power (dBm)", 0, 40, 10)
rx_sens = st.sidebar.number_input("RX Sensitivity (dBm)", value=-94)
fade_margin = st.sidebar.slider("Fade Margin (dB)", 0, 20, 10)
required_rssi = rx_sens + fade_margin
map_width_m = st.sidebar.number_input("Map Width (m)", value=50.0)
wall_loss_sens = st.sidebar.slider("Wall Loss Sens.", 0, 20, 8)

uploaded_file = st.sidebar.file_uploader("Upload Map", type=['png', 'jpg'])

if uploaded_file:
    img_pil = Image.open(uploaded_file).convert("RGB")
    w, h = img_pil.size
    px_to_m = map_width_m / w
    
    img_np = np.array(img_pil)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, wall_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    col_map, col_ctrl = st.columns([0.85, 0.15])

    with col_ctrl:
        st.subheader("Mode")
        mode = st.radio("Action:", ["Add TX", "Add RP", "Add RX", "Remove"])
        if st.button("Clear All"):
            st.session_state.devices = []
            st.rerun()

    with col_map:
        draw_img = img_pil.copy()
        sources = [d for d in st.session_state.devices if d['type'] in ['TX', 'RP']]
        
        if sources:
            # --- 픽셀 촘촘함 수정 부분 ---
            res = 8 # 기존 20에서 8로 변경 (값이 작을수록 더 촘촘함)
            # ---------------------------
            overlay = Image.new('RGBA', (w, h), (0,0,0,0))
            ov_draw = ImageDraw.Draw(overlay)
            for y in range(0, h, res):
                for x in range(0, w, res):
                    max_rssi = -150.0
                    for s in sources:
                        dist = np.sqrt((x-s['x'])**2 + (y-s['y'])**2) * px_to_m
                        nw = count_walls_fast((s['x'], s['y']), (x, y), wall_mask)
                        pwr = tx_eff if s['type'] == 'TX' else rp_eff
                        rssi = pwr - calculate_fspl(max(dist, 0.5)) - (nw * wall_loss_sens)
                        if rssi > max_rssi: max_rssi = rssi
                    if max_rssi >= required_rssi - 15:
                        norm = np.clip((max_rssi - rx_sens) / (max(tx_eff, rp_eff) - rx_sens), 0, 1)
                        c = cm.jet(norm)
                        # 투명도를 살짝 조절하여 촘촘할 때 답답하지 않게 설정
                        ov_draw.rectangle([x, y, x+res, y+res], fill=(int(c[0]*255), int(c[1]*255), int(c[2]*255), 130))
            draw_img.paste(overlay, (0,0), overlay)

        draw = ImageDraw.Draw(draw_img)
        for i, d in enumerate(st.session_state.devices):
            r = 15
            color = (255, 0, 0) if d['type'] == 'TX' else ((255, 165, 0) if d['type'] == 'RP' else (0, 0, 255))
            if d['type'] == 'RX': draw.rectangle([d['x']-r, d['y']-r, d['x']+r, d['y']+r], fill=color, outline="white", width=3)
            else: draw.ellipse([d['x']-r, d['y']-r, d['x']+r, d['y']+r], fill=color, outline="white", width=3)
            draw.text((d['x']+18, d['y']-10), f"{d['type']}{i+1}", fill="white")

        clicked = streamlit_image_coordinates(draw_img, width=1200, key="map_main")

        if clicked and clicked != st.session_state.last_click:
            st.session_state.last_click = clicked
            cx, cy = clicked['x'], clicked['y']
            scale = w / 1200
            real_x, real_y = cx * scale, cy * scale
            
            if "TX" in mode: st.session_state.devices.append({'type': 'TX', 'x': real_x, 'y': real_y})
            elif "RP" in mode: st.session_state.devices.append({'type': 'RP', 'x': real_x, 'y': real_y})
            elif "RX" in mode: st.session_state.devices.append({'type': 'RX', 'x': real_x, 'y': real_y})
            elif "Remove" in mode:
                st.session_state.devices = [d for d in st.session_state.devices if np.sqrt((d['x']-real_x)**2 + (d['y']-real_y)**2) > 40]
            st.rerun()

        st.write("### Signal Intensity Guide")
        fig_leg, ax_leg = plt.subplots(figsize=(12, 0.8))
        max_p = max(tx_eff, rp_eff)
        gradient = np.linspace(rx_sens - 15, max_p, 256).reshape(1, 256)
        ax_leg.imshow(gradient, aspect='auto', cmap='jet', extent=[rx_sens - 15, max_p, 0, 1])
        ax_leg.set_yticks([])
        ax_leg.set_xlabel("Signal Strength (dBm)")
        ax_leg.axvspan(rx_sens - 15, required_rssi, color='black', alpha=0.4)
        ax_leg.text((rx_sens - 15 + required_rssi)/2, 0.5, "FAIL ZONE", color='white', ha='center', va='center', fontweight='bold')
        ax_leg.axvline(required_rssi, color='red', linestyle='--', linewidth=2)
        st.pyplot(fig_leg)

    # Report
    rxs = [d for d in st.session_state.devices if d['type'] == 'RX']
    if rxs and sources:
        st.divider()
        st.subheader("📊 Signal Analysis Report")
        cols = st.columns(len(rxs) if len(rxs) < 6 else 6)
        for i, rx in enumerate(rxs):
            best_r = -150.0
            for s in sources:
                d_m = np.sqrt((rx['x']-s['x'])**2 + (rx['y']-s['y'])**2) * px_to_m
                nw = count_walls_fast((s['x'], s['y']), (rx['x'], rx['y']), wall_mask)
                pwr = tx_eff if s['type'] == 'TX' else rp_eff
                r = pwr - calculate_fspl(max(d_m, 0.5)) - (nw * wall_loss_sens)
                if r > best_r: best_r = r
            with cols[i % 6]:
                st.metric(f"RX {i+1}", f"{best_r:.1f} dBm", delta="OK" if best_r >= required_rssi else "FAIL")
else:
    st.info("Please upload a floor plan to start.")