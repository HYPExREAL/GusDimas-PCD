import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import time

st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    * { 
        font-family: 'Plus Jakarta Sans', sans-serif !important; 
    }
    
    /* Ultra Clean Background */
    .stApp { 
        background: #F8F9FA;
    }
    
    .block-container { 
        padding: 2rem 3rem !important;
        max-width: 1500px !important; 
    }
    
    #MainMenu, footer, header { visibility: hidden; }
    
    /* SLEEK HEADER */
    .app-header {
        background: white;
        padding: 1.5rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
        border-left: 4px solid #F97316;
    }
    
    .brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .brand-icon {
        font-size: 2rem;
    }
    
    .brand-text {
        font-size: 1.875rem;
        font-weight: 800;
        color: #111827;
        letter-spacing: -0.02em;
    }
    
    .brand-accent {
        color: #F97316;
    }
    
    .brand-tagline {
        font-size: 0.875rem;
        color: #6B7280;
        font-weight: 500;
        margin-top: 0.125rem;
    }
    

    
    /* PREMIUM CARDS */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        border: 1px solid #F0F0F0;
        transition: all 0.2s ease;
    }
    
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"]:hover {
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1rem;
        font-weight: 700;
        color: #374151;
        margin-bottom: 1.5rem;
        padding-bottom: 0.875rem;
        border-bottom: 2px solid #F3F4F6;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .section-icon {
        font-size: 1.25rem;
    }
    
    /* ELEGANT FILE UPLOADER */
    [data-testid="stFileUploader"] {
        border: 2px dashed #D1D5DB;
        border-radius: 12px;
        padding: 2.5rem 1.5rem;
        background: #FAFAFA;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #F97316;
        background: #FFF7ED;
        transform: translateY(-2px);
    }
    
    [data-testid="stFileUploader"] section {
        border: none !important;
    }
    
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #F97316 0%, #EA580C 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.625rem 1.75rem !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        box-shadow: 0 2px 4px rgba(249, 115, 22, 0.2) !important;
        transition: all 0.2s !important;
    }
    
    [data-testid="stFileUploader"] button:hover {
        box-shadow: 0 4px 8px rgba(249, 115, 22, 0.3) !important;
        transform: translateY(-1px) !important;
    }
    
    /* POWERFUL BUTTON */
    .stButton > button {
        background: linear-gradient(135deg, #F97316 0%, #EA580C 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 1rem 2rem !important;
        border-radius: 12px !important;
        width: 100% !important;
        box-shadow: 0 4px 6px rgba(249, 115, 22, 0.25) !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #EA580C 0%, #DC2626 100%) !important;
        box-shadow: 0 6px 12px rgba(249, 115, 22, 0.4) !important;
        transform: translateY(-2px) scale(1.01);
    }
    
    .stButton > button:active {
        transform: translateY(0) scale(0.99) !important;
    }
    
    /* BEAUTIFUL METRICS */
    [data-testid="stMetricValue"] {
        font-size: 2.25rem !important;
        font-weight: 800 !important;
        color: #111827 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        color: #6B7280 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #FAFAFA 0%, #F3F4F6 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        transition: all 0.2s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* CRISP IMAGES */
    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #E5E7EB;
    }
    
    [data-testid="stImage"] img {
        border-radius: 12px !important;
    }
    
    /* POLISHED ALERTS */
    div[data-baseweb="notification"] {
        border-radius: 12px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 1.125rem 1.5rem !important;
        font-size: 0.938rem !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06) !important;
    }
    
    div[data-baseweb="notification"][kind="success"] {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%) !important;
        color: #065F46 !important;
        border-left: 4px solid #10B981 !important;
    }
    
    div[data-baseweb="notification"][kind="error"] {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%) !important;
        color: #991B1B !important;
        border-left: 4px solid #EF4444 !important;
    }
    
    div[data-baseweb="notification"][kind="warning"] {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%) !important;
        color: #92400E !important;
        border-left: 4px solid #F59E0B !important;
    }
    
    div[data-baseweb="notification"][kind="info"] {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%) !important;
        color: #1E40AF !important;
        border-left: 4px solid #3B82F6 !important;
    }
    
    /* SMOOTH PROGRESS BAR */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #F97316 0%, #EA580C 100%) !important;
        border-radius: 8px !important;
    }
    
    .stProgress > div > div {
        background: #F3F4F6 !important;
        border-radius: 8px !important;
    }
    
    /* REFINED EXPANDER */
    div[data-testid="stExpander"] {
        background: #FAFAFA !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.2s ease !important;
        overflow: hidden !important;
    }
    
    div[data-testid="stExpander"]:hover {
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08) !important;
    }
    
    div[data-testid="stExpander"] details {
        background: transparent !important;
    }
    
    div[data-testid="stExpander"] summary {
        font-weight: 600 !important;
        color: #374151 !important;
        font-size: 0.938rem !important;
        padding: 1rem 1.25rem !important;
        cursor: pointer !important;
        list-style: none !important;
        background: transparent !important;
    }
    
    div[data-testid="stExpander"] summary::-webkit-details-marker {
        display: none !important;
    }
    
    div[data-testid="stExpander"] summary::before {
        content: "▶" !important;
        display: inline-block !important;
        margin-right: 0.5rem !important;
        transition: transform 0.2s !important;
        font-size: 0.75rem !important;
    }
    
    div[data-testid="stExpander"] details[open] summary::before {
        transform: rotate(90deg) !important;
    }
    
    div[data-testid="stExpander"] details[open] summary {
        border-bottom: 1px solid #E5E7EB !important;
        margin-bottom: 0 !important;
        padding-bottom: 1rem !important;
    }
    
    div[data-testid="stExpander"] details[open] > div {
        padding: 1rem 1.25rem !important;
    }
    
    /* DIVIDER */
    hr {
        margin: 2rem 0 !important;
        border: none !important;
        border-top: 1px solid #E5E7EB !important;
    }
    
    /* EMPTY STATES */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        background: linear-gradient(135deg, #FAFAFA 0%, #F3F4F6 100%);
        border-radius: 12px;
        border: 2px dashed #D1D5DB;
    }
    
    .empty-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.4;
        filter: grayscale(100%);
    }
    
    .empty-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #4B5563;
        margin-bottom: 0.5rem;
    }
    
    .empty-text {
        font-size: 0.938rem;
        color: #9CA3AF;
        font-weight: 500;
    }
    
    /* Image Info Badge */
    .image-info {
        background: #F3F4F6;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        text-align: center;
        margin-top: 1rem;
        font-size: 0.813rem;
        color: #6B7280;
        font-weight: 600;
    }
    
    /* Subtle animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.4s ease-out;
    }
    
    /* Results container */
    .results-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

MODEL_PATH = "runs/detect/train6/weights/best.pt"
CONF_THRESHOLD = 0.45

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

def pcd_pipeline(img_rgb):
    """Preprocessing: Grayscale + CLAHE + Gaussian Blur"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    return cv2.merge([blurred, blurred, blurred])

try:
    model = load_model()
    model_loaded = True
    model_status = "active"
except Exception as e:
    model_loaded = False
    model_status = "error"

st.markdown("""
<div class="app-header">
    <div class="brand">
        <div class="brand-icon">🛡️</div>
        <div>
            <div class="brand-text">Deteksi Penggunaan Helm<span class="brand-accent"> App</span></div>
            <div class="brand-tagline">Sistem Deteksi Helm Pengendara berbasis Pengolahan Citra Digital dengan YOLOv11</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col_upload, col_results = st.columns([1, 1.6], gap="large")

with col_upload:
    st.markdown('<div class="section-header"><span class="section-icon">📤</span> Unggah Gambar</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Letakkan gambar Anda di sini",
        type=['jpg', 'png', 'jpeg'],
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_rgb = np.array(image)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        
        file_size = uploaded_file.size / 1024  # KB
        st.markdown(f'<div class="image-info">📁 {uploaded_file.name} · {file_size:.1f} KB · {img_rgb.shape[1]}x{img_rgb.shape[0]}px</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if model_loaded:
            analyze_btn = st.button("🚀 Mulai Deteksi", use_container_width=True)
        else:
            st.error("⚠️ Model tidak tersedia. Periksa MODEL_PATH.")
            analyze_btn = False
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📷</div>
            <div class="empty-title">Belum Ada Gambar</div>
            <div class="empty-text">Unggah gambar untuk memulai deteksi helm</div>
        </div>
        """, unsafe_allow_html=True)
        analyze_btn = False

with col_results:
    st.markdown('<div class="section-header"><span class="section-icon">📊</span> Hasil Deteksi</div>', unsafe_allow_html=True)
    
    if analyze_btn and uploaded_file and model_loaded:
        progress_bar = st.progress(0)
        status_msg = st.empty()
        
        status_msg.info("🔄 Menerapkan preprocessing gambar...")
        img_in = pcd_pipeline(img_rgb)
        progress_bar.progress(30)
        time.sleep(0.35)
        
        status_msg.info("🔍 Menjalankan deteksi YOLOv11...")
        results = model(img_in, conf=CONF_THRESHOLD)
        progress_bar.progress(70)
        time.sleep(0.35)
        
        status_msg.info("✨ Membuat visualisasi...")
        img_vis = img_rgb.copy()
        stats = {'safe': 0, 'danger': 0}
        detections = []
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                
                is_danger = any(word in label.lower() for word in ["tanpa", "no", "tidak", "without"])
                
                if is_danger:
                    color = (239, 68, 68)
                    stats['danger'] += 1
                    category = "danger"
                else:
                    color = (16, 185, 129)
                    stats['safe'] += 1
                    category = "safe"
                
                detections.append({
                    'label': label,
                    'confidence': conf,
                    'category': category
                })
                
                thickness = 3
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, thickness)
                
                label_text = label
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                
                (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
                
                cv2.rectangle(img_vis, 
                            (x1, y1 - text_h - baseline - 10), 
                            (x1 + text_w + 10, y1), 
                            color, -1)
                
                cv2.putText(img_vis, label_text, 
                           (x1 + 5, y1 - baseline - 5), 
                           font, font_scale, (255, 255, 255), font_thickness)
        
        progress_bar.progress(100)
        time.sleep(0.25)
        progress_bar.empty()
        status_msg.empty()
        
        st.image(img_vis, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        total_detections = stats['safe'] + stats['danger']
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("Total Terdeteksi", total_detections, help="Jumlah total pengendara yang terdeteksi")
        
        with metric_col2:
            st.metric("Pakai Helm ✓", stats['safe'], help="Pengendara yang memakai helm")
        
        with metric_col3:
            st.metric("Tidak Pakai Helm ✗", stats['danger'], delta=f"-{stats['danger']}" if stats['danger'] > 0 else None, delta_color="inverse", help="Pengendara yang tidak memakai helm")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if stats['danger'] > 0:
            danger_pct = (stats['danger'] / total_detections * 100) if total_detections > 0 else 0
            st.error(f"⚠️ **Pelanggaran Keselamatan Terdeteksi!** {stats['danger']} pengendara tidak memakai helm ({danger_pct:.0f}% tingkat pelanggaran)", icon="🚨")
        elif stats['safe'] > 0:
            st.success(f"✅ **Semua Aman!** Semua {stats['safe']} pengendara yang terdeteksi memakai helm dengan benar.", icon="🛡️")
        else:
            st.warning("ℹ️ Tidak ada pengendara yang terdeteksi pada gambar ini. Coba unggah gambar lain.", icon="🔍")
    
    else:
        st.markdown("""
        <div class="empty-state" style="min-height: 480px; display: flex; flex-direction: column; justify-content: center;">
            <div class="empty-icon">🛡️</div>
            <div class="empty-title">Siap Menganalisis</div>
            <div class="empty-text">Unggah gambar dan klik "Mulai Deteksi" untuk memulai analisis</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #9CA3AF; font-size: 0.875rem; padding: 1rem;">
    <div style="font-weight: 600; color: #6B7280; margin-bottom: 0.25rem;">Sistem Deteksi Penggunaan Helm</div>
    <div>Model - YOLOv11 · Confidence Threshold: 45% · Preprocessing - Grayscale + CLAHE + Gaussian Blur</div>
    <div>© Gus Dimas PCD</div>
</div>
""", unsafe_allow_html=True)