import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# --- 1. Physics Model (Realistic Indoor Path Loss) ---
FREQUENCY_MHZ = 447
FLOOR_HEIGHT_M = 4.0

def calculate_indoor_path_loss(distance_m):
    """
    실내 환경을 고려한 Log-Distance Path Loss 모델.
    자유공간보다 거리에 따른 감쇄 폭이 훨씬 큼.
    """
    if distance_m <= 1.0: 
        # 1m 이내 근접 거리는 기본 자유공간 공식 적용
        return 20 * np.log10(1.0) + 20 * np.log10(FREQUENCY_MHZ) - 27.55
    
    # 실내 환경 감쇄 계수 (n=3.0 ~ 3.5 적용)
    # 일반적인 사무실/건물 내부는 n=3.0 이상일 때 거리에 따른 감쇄가 뚜렷함
    n = 3.2 
    reference_loss = 20 * np.log10(FREQUENCY_MHZ) - 27.55 # 1m 기준 손실
    return reference_loss + 10 * n * np.log10(distance_m)

def count_walls_fast(start_pt, end_pt, wall_mask, step=10):
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

# --- 2. Interface Setup ---
st.set_page_config(layout="wide", page_title="Wireless Signal Map Simulator")

st.markdown("""
    <style>
    section[data-testid="stSidebar"] { width: 320px !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🏢 Wireless Signal Map Simulator")

if 'devices' not in st.session_state: st.session_state.devices = []
if 'last_click' not in st.session_state: st.session_state.last_click = {}

# --- 3. Sidebar (Controls placed right above Spectrum) ---
st.sidebar.header("⚙️ Settings")
tx_eff = st.sidebar.slider("TX Power (dBm)", -10, 20, 0)
rp_eff = st.sidebar.slider("RP Power (dBm)", 0, 40, 10)
rx_sens = st.sidebar.number_input("RX Sensitivity (dBm)", value=-94)
fade_margin = st.sidebar.slider("Fade Margin (dB)", 0, 20, 10)
required_rssi = rx_sens + fade_margin
slab_loss_db = st.sidebar.slider("Slab Loss (dB/floor)", 10, 50, 20)
wall_loss_sens = st.sidebar.slider("Wall Loss Sens.", 0, 20, 5)
map_width_m = st.sidebar.number_input("Map Width (m)", value=50.0)

st.sidebar.write("---")
# 요청사항: Controls 항목을 스펙트럼 바로 위에 배치
st.sidebar.subheader("🕹️ Controls")
mode = st.sidebar.radio("Action Mode:", ["Add TX", "Add RP", "Add RX", "Remove"])
if st.sidebar.button("Clear All Devices"):
    st.session_state.devices = []
    st.rerun()

# Spectrum Visualization
st.sidebar.subheader("📊 Signal Strength Guide")
fig_leg, ax_leg = plt.subplots(figsize=(1.2, 5))
max_p = max(tx_eff, rp_eff)
gradient = np.linspace(max_p, rx_sens - 15, 256).reshape(256, 1)
ax_leg.imshow(gradient, aspect='auto', cmap='jet', extent=[0, 1, rx_sens - 15, max_p])
ax_leg.yaxis.tick_right()
ax_leg.yaxis.set_label_position("right")
ax_leg.set_xticks([])
ax_leg.set_ylabel("RSSI (dBm)", fontsize=9)
ax_leg.axhspan(rx_sens - 15, required_rssi, color='black', alpha=0.5)
ax_leg.text(0.5, (rx_sens - 15 + required_rssi)/2, "FAIL ZONE", color='white', ha='center', va='center', fontweight='bold', rotation=90)
ax_leg.axhline(required_rssi, color='red', linestyle='--', linewidth=1.5)
st.sidebar.pyplot(fig_leg)

# --- 4. Main Display ---
uploaded_files = st.file_uploader("Upload Floor Plans", type=['png', 'jpg'], accept_multiple_files=True)

if uploaded_files:
    sorted_files = sorted(uploaded_files, key=lambda x: x.name, reverse=True)
    floor_images, floor_masks, floor_names = [], [], []
    
    for file in sorted_files:
        img = Image.open(file).convert("RGB")
        floor_images.append(img)
        floor_names.append(file.name)
        gnp = np.array(img)
        gray = cv2.cvtColor(gnp, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        floor_masks.append(mask)

    for f_idx in range(len(floor_images)):
        st.divider()
        st.subheader(f"🏢 {floor_names[f_idx]}")
        
        img_w, img_h = floor_images[f_idx].size
        px_to_m = map_width_m / img_w 
        
        draw_img = floor_images[f_idx].copy()
        sources = [d for d in st.session_state.devices if d['type'] in ['TX', 'RP']]
        
        if sources:
            res = 12 
            overlay = Image.new('RGBA', (img_w, img_h), (0,0,0,0))
            ov_draw = ImageDraw.Draw(overlay)
            for y in range(0, img_h, res):
                for x in range(0, img_w, res):
                    max_rssi = -150.0
                    for s in sources:
                        dx_m = (x - s['x']) * px_to_m
                        dy_m = (y - s['y']) * px_to_m
                        h_dist_m = np.sqrt(dx_m**2 + dy_m**2)
                        v_dist_m = abs(f_idx - s['floor_idx']) * FLOOR_HEIGHT_M
                        total_dist_m = np.sqrt(h_dist_m**2 + v_dist_m**2)
                        
                        pwr = tx_eff if s['type'] == 'TX' else rp_eff
                        # 수정된 실내 전파 감쇄 모델 적용
                        rssi = pwr - calculate_indoor_path_loss(total_dist_m)
                        
                        rssi -= (abs(f_idx - s['floor_idx']) * slab_loss_db)
                        nw = count_walls_fast((s['x'], s['y']), (x, y), floor_masks[f_idx])
                        rssi -= (nw * wall_loss_sens)
                        
                        if rssi > max_rssi: max_rssi = rssi
                            
                    if max_rssi >= required_rssi - 15:
                        norm = np.clip((max_rssi - rx_sens) / (max_p - rx_sens), 0, 1)
                        c = cm.jet(norm)
                        ov_draw.rectangle([x, y, x+res, y+res], fill=(int(c[0]*255), int(c[1]*255), int(c[2]*255), 110))
            draw_img.paste(overlay, (0,0), overlay)

        draw = ImageDraw.Draw(draw_img)
        for i, d in enumerate(st.session_state.devices):
            if d['floor_idx'] == f_idx:
                r = 15
                color = (255, 0, 0) if d['type'] == 'TX' else ((255, 165, 0) if d['type'] == 'RP' else (0, 0, 255))
                if d['type'] == 'RX': draw.rectangle([d['x']-r, d['y']-r, d['x']+r, d['y']+r], fill=color, outline="white", width=3)
                else: draw.ellipse([d['x']-r, d['y']-r, d['x']+r, d['y']+r], fill=color, outline="white", width=3)
                draw.text((d['x']+18, d['y']-10), f"{d['type']}", fill="white", stroke_fill="black", stroke_width=1)

        clicked = streamlit_image_coordinates(draw_img, width=1200, key=f"map_{f_idx}")

        if clicked and clicked != st.session_state.last_click.get(f_idx):
            st.session_state.last_click[f_idx] = clicked
            scale = img_w / 1200
            rx, ry = clicked['x'] * scale, clicked['y'] * scale
            if "Add" in mode:
                st.session_state.devices.append({'type': mode.split()[1], 'x': rx, 'y': ry, 'floor_idx': f_idx})
            elif "Remove" in mode:
                st.session_state.devices = [d for d in st.session_state.devices if not (d['floor_idx'] == f_idx and np.sqrt((d['x']-rx)**2 + (d['y']-ry)**2) < 40)]
            st.rerun()

    # Analysis Report
    st.divider()
    rxs = [d for d in st.session_state.devices if d['type'] == 'RX']
    if rxs and sources:
        st.subheader("📊 Signal Analysis Report")
        cols = st.columns(min(len(rxs), 5))
        for i, rx in enumerate(rxs):
            best_r = -150.0
            for s in sources:
                dx = (rx['x']-s['x']) * px_to_m
                dy = (rx['y']-s['y']) * px_to_m
                h_d = np.sqrt(dx**2 + dy**2)
                v_d = abs(rx['floor_idx'] - s['floor_idx']) * FLOOR_HEIGHT_M
                t_d = np.sqrt(h_d**2 + v_d**2)
                # 수치 분석 리포트에도 실내 모델 적용
                r = (tx_eff if s['type'] == 'TX' else rp_eff) - calculate_indoor_path_loss(t_d)
                r -= (abs(rx['floor_idx'] - s['floor_idx']) * slab_loss_db)
                r -= (count_walls_fast((s['x'], s['y']), (rx['x'], rx['y']), floor_masks[rx['floor_idx']]) * wall_loss_sens)
                if r > best_r: best_r = r
            with cols[i % 5]:
                st.metric(f"RX {i+1} ({floor_names[rx['floor_idx']]})", f"{best_r:.1f} dBm", delta="OK" if best_r >= required_rssi else "FAIL")
else:
    st.info("Upload floor plans to start simulation.")
