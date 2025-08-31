import streamlit as st
import numpy as np
import cv2
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

st.set_page_config(page_title="Streamlit Image Lab", layout="wide")

st.title("Streamlit Image Lab — Webcam / URL / Upload + Simple Image Processing")

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("1) แหล่งที่มาของภาพ")
source = st.sidebar.selectbox(
    "เลือกแหล่งภาพ",
    ["Webcam (ถ่ายภาพ)", "อัปโหลดไฟล์", "Image URL"],
    help="เลือกว่าจะใช้กล้องโน้ตบุ๊ค, อัปโหลดภาพ, หรืออ่านจากลิงก์ URL"
)

st.sidebar.header("2) การประมวลผลภาพ")
operation = st.sidebar.selectbox(
    "เลือกกระบวนการ (เลือก 1 อย่าง)",
    ["Canny Edge Detection", "Binary Threshold", "CLAHE (เพิ่มคอนทราสต์)"]
)

scale_pct = st.sidebar.slider("อัตราการย่อ/ขยาย (%)", 10, 200, 100, 5, help="ปรับขนาดภาพก่อนประมวลผล")

# Parameters per operation
if operation == "Canny Edge Detection":
    st.sidebar.subheader("พารามิเตอร์ Canny")
    low_thresh = st.sidebar.slider("Threshold 1", 0, 255, 100, 1)
    high_thresh = st.sidebar.slider("Threshold 2", 0, 255, 200, 1)
    aperture = st.sidebar.selectbox("Aperture Size (Sobel)", [3, 5, 7], index=0)
    l2grad = st.sidebar.checkbox("ใช้ L2gradient", value=True)
elif operation == "Binary Threshold":
    st.sidebar.subheader("พารามิเตอร์ Threshold")
    th_mode = st.sidebar.radio("โหมด", ["Manual", "Otsu"], index=0)
    th_val = st.sidebar.slider("Threshold (Manual)", 0, 255, 128, 1, disabled=(th_mode=="Otsu"))
    invert = st.sidebar.checkbox("Invert (ขาว-ดำสลับ)", value=False)
elif operation == "CLAHE (เพิ่มคอนทราสต์)":
    st.sidebar.subheader("พารามิเตอร์ CLAHE")
    clip = st.sidebar.slider("clipLimit", 1.0, 8.0, 3.0, 0.1)
    tiles = st.sidebar.slider("tileGridSize", 2, 16, 8, 1)

st.sidebar.header("3) ตัวเลือกทั่วไป")
show_hist_on = st.sidebar.checkbox("แสดง Histogram (กราฟ)", value=True)
download_on = st.sidebar.checkbox("แสดงปุ่มดาวน์โหลดภาพผลลัพธ์", value=True)


# ----------------------------
# Helper functions
# ----------------------------
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image (RGB) to OpenCV BGR numpy array."""
    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr

def resize_by_percent(img_bgr: np.ndarray, pct: int) -> np.ndarray:
    if pct == 100:
        return img_bgr
    h, w = img_bgr.shape[:2]
    new_w = max(1, int(w * pct / 100.0))
    new_h = max(1, int(h * pct / 100.0))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA if pct < 100 else cv2.INTER_LINEAR)

def load_from_url(url: str) -> Image.Image:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return Image.open(BytesIO(r.content))

def to_png_bytes(img_bgr: np.ndarray) -> bytes:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

def compute_hist_fig(gray: np.ndarray):
    fig, ax = plt.subplots()
    ax.hist(gray.ravel(), bins=256, range=(0, 256))
    ax.set_title("Grayscale Histogram")
    ax.set_xlabel("Intensity (0-255)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig

def apply_processing(bgr: np.ndarray):
    """Return processed_bgr, metrics_dict, fig_hist"""
    # Resize first
    bgr = resize_by_percent(bgr, scale_pct)

    # Common grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Process by operation
    metrics = {}
    if operation == "Canny Edge Detection":
        edges = cv2.Canny(gray, low_thresh, high_thresh, apertureSize=aperture, L2gradient=l2grad)
        # Overlay edges (red) on original color
        overlay = bgr.copy()
        overlay[edges > 0] = [0, 0, 255]  # red in BGR
        processed = overlay
        edge_ratio = float(np.count_nonzero(edges)) / edges.size
        metrics["Edge pixels (%)"] = round(100.0 * edge_ratio, 2)
    elif operation == "Binary Threshold":
        if th_mode == "Otsu":
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, bw = cv2.threshold(gray, th_val, 255, cv2.THRESH_BINARY)
        if invert:
            bw = cv2.bitwise_not(bw)
        processed = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        white_ratio = float(np.count_nonzero(bw == 255)) / bw.size
        metrics["White area (%)"] = round(100.0 * white_ratio, 2)
        if th_mode == "Otsu":
            metrics["Method"] = "Otsu's thresholding"
        else:
            metrics["Threshold"] = th_val
    else:  # CLAHE
        clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tiles), int(tiles)))
        eq = clahe.apply(gray)
        processed = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
        metrics["CLAHE clipLimit"] = clip
        metrics["CLAHE tileGrid"] = f"{tiles}x{tiles}"

    # General metrics
    metrics["Width x Height"] = f"{bgr.shape[1]} x {bgr.shape[0]} px"
    metrics["Mean intensity (gray)"] = round(float(gray.mean()), 2)
    metrics["Std intensity (gray)"] = round(float(gray.std()), 2)

    hist_fig = compute_hist_fig(gray) if show_hist_on else None
    return processed, metrics, hist_fig


# ----------------------------
# Load image based on source
# ----------------------------
image_bgr = None
error = None

if source == "Webcam (ถ่ายภาพ)":
    cam_img = st.camera_input("ถ่ายภาพจากกล้องโน้ตบุ๊คแล้วกด Capture")
    if cam_img is not None:
        pil_img = Image.open(cam_img)
        image_bgr = pil_to_bgr(pil_img)

elif source == "อัปโหลดไฟล์":
    up = st.file_uploader("อัปโหลดภาพ (PNG/JPG/JPEG/BMP)", type=["png", "jpg", "jpeg", "bmp"])
    if up is not None:
        pil_img = Image.open(up)
        image_bgr = pil_to_bgr(pil_img)

else:  # Image URL
    url = st.text_input("วางลิงก์รูปภาพ (เช่น https://.../image.jpg)")
    if url:
        try:
            pil_img = load_from_url(url)
            image_bgr = pil_to_bgr(pil_img)
        except Exception as e:
            error = str(e)

if error:
    st.error(f"ไม่สามารถโหลดรูปจาก URL ได้: {error}")

if image_bgr is None:
    st.info("กรุณาเลือก/ถ่าย/ใส่ลิงก์รูปภาพทางซ้าย แล้วรูปจะถูกแสดงและประมวลผลที่นี่")
    st.stop()

# ----------------------------
# Process + Display
# ----------------------------
proc_bgr, metrics, hist_fig = apply_processing(image_bgr)

c1, c2 = st.columns(2, gap="large")
with c1:
    st.subheader("ภาพต้นฉบับ")
    st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

with c2:
    st.subheader("ผลลัพธ์หลังประมวลผล")
    st.image(cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

st.subheader("สถิติของภาพ")
m1, m2, m3 = st.columns(3)
items = list(metrics.items())
for idx, (k, v) in enumerate(items):
    if idx % 3 == 0:
        with m1: st.metric(k, v)
    elif idx % 3 == 1:
        with m2: st.metric(k, v)
    else:
        with m3: st.metric(k, v)

if hist_fig is not None:
    st.subheader("กราฟจากคุณสมบัติของภาพ (Histogram ความเข้มแสง)")
    st.pyplot(hist_fig, clear_figure=True)

if download_on:
    st.download_button(
        "ดาวน์โหลดภาพผลลัพธ์ (PNG)",
        data=to_png_bytes(proc_bgr),
        file_name="processed.png",
        mime="image/png"
    )

st.caption("หมายเหตุ: โค้ดนี้ใช้ OpenCV, Pillow, NumPy, Matplotlib และ Streamlit")