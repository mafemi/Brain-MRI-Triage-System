import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms as T
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import time
import base64
import os

# Try importing from pytorch_grad_cam, provide fallbacks if missing
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    HAS_GRAD_CAM = True
except ImportError:
    HAS_GRAD_CAM = False

### ==========================================
### 1. PAGE CONFIGURATION & SETUP
### ==========================================
st.set_page_config(page_title="Brain MRI Triage System", page_icon="🧠", layout="wide")

# Custom CSS overrides for horizontal alignment, narrowing sidepanel, and styling sleek premium navigation buttons
st.markdown("""
    <style>
    /* Align the top of the main screen perfectly with the sidebar */
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Remove native margin and padding from the main page h1 header */
    .block-container h1 {
        margin-top: 0px !important;
        padding-top: 0px !important;
        margin-bottom: 0.5rem !important;
        line-height: 1.2 !important;
    }
    
    /* Set narrow, professional width for the sidebar to exactly fit 200px logo + 0.1cm left/right padding */
    section[data-testid="stSidebar"] {
        min-width: calc(200px + 0.5cm) !important;
        max-width: calc(200px + 1cm) !important;
    }
    
    /* Reduce sidebar content area padding to exactly 0.1cm horizontally and 3.5rem on top to match main page */
    [data-testid="stSidebarUserContent"] {
        padding-left: 0.1cm !important;
        padding-right: 0.1cm !important;
        padding-top: 0rem !important; /* perfectly matches main container padding */
    }
    
    /* Style the sidebar radio selections into clean, compact SaaS-grade navigation buttons */
    div[role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 8px !important;
        padding-top: 5px !important;
    }
    
    /* Force radio labels to stretch as complete sleek button items */
    div[role="radiogroup"] label {
        background-color: rgba(128, 128, 128, 0.05) !important;
        border: 1px solid rgba(128, 128, 128, 0.12) !important;
        border-radius: 8px !important;
        padding: 5px 5px !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        cursor: pointer !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    /* Hide native circular radio check dots completely */
    div[role="radiogroup"] label [data-testid="stRadioButtonHoverTarget"],
    div[role="radiogroup"] label div[role="presentation"] {
        display: none !important;
    }
    
    /* Ensure the tab labels are larger, premium, bold buttons to match the box size beautifully */
    div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
        font-size: 1.05rem !important; /* increased to match the box size beautifully */
        font-weight: 500 !important;
        margin: 0 !important;
        padding: 0 !important;
        color: var(--text-color) !important;
    }
    
    /* Elegant hover states for buttons */
    div[role="radiogroup"] label:hover {
        background-color: rgba(79, 70, 229, 0.08) !important;
        border-color: rgba(79, 70, 229, 0.25) !important;
    }
    div[role="radiogroup"] label:hover p {
        color: #4F46E5 !important;
    }
    
    /* Active selected button styling using clinical academic violet background */
    div[role="radiogroup"] label[data-checked="true"] {
        background-color: #4F46E5 !important;
        border-color: #4F46E5 !important;
        box-shadow: 0 2px 5px rgba(79, 70, 229, 0.3) !important;
    }
    
    div[role="radiogroup"] label[data-checked="true"] p {
        color: white !important;
        font-weight: 600 !important;
    }

    
    /* Logo container with strict 0.1cm horizontal spacing and absolute top alignment */
    .logo-container {
        text-align: center !important;
        margin-bottom: 15px !important;
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
        padding-left: 0.1cm !important;
        padding-right: 0.1cm !important;
        margin-top: 0px !important;
        padding-top: 0px !important;
    }
    .logo-img {
        width: 200px !important;
        height: auto !important;
        display: block !important;
        mix-blend-mode: difference !important;
    }
    </style>

""", unsafe_allow_html=True)

# Standardise class names with UK English "No Tumour" to match the thesis
CLASSES = ['Glioma', 'Meningioma', 'No Tumour', 'Pituitary']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### ==========================================
### 2. DATA PREPROCESSING PIPELINE
### ==========================================
class StrictBrainMaskAndCrop(object):
    def __call__(self, img):
        img_np = np.array(img)
        # Handle colour space standardisation
        if len(img_np.shape) == 2:
            gray = img_np
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        else:
            img_rgb = img_np
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
        # Stage 1: Noise Smoothing
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Stage 2: Automatic Thresholding (Otsu's Binarisation)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Stage 3: Anatomical Reconstruction (Morphological Closing)
        kernel = np.ones((7, 7), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Stage 4: Contour Tracking & Mask AND Operation
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img
        c = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [c], -1, 255, -1)
        img_masked = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
        
        # Calculate extreme points with a 10px safety margin
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        
        margin = 10
        ymin = max(0, extTop[1] - margin)
        ymax = min(img_masked.shape[0], extBot[1] + margin)
        xmin = max(0, extLeft[0] - margin)
        xmax = min(img_masked.shape[1], extRight[0] + margin)
        
        cropped_img = img_masked[ymin:ymax, xmin:xmax]
        if cropped_img.size == 0:
            return img
            
        # Stage 5: Symmetrical Aspect-Ratio Preserving Square Padding
        h, w = cropped_img.shape[:2]
        max_side = max(h, w)
        top = (max_side - h) // 2
        bottom = max_side - h - top
        left = (max_side - w) // 2
        right = max_side - w - left
        
        squared_img = cv2.copyMakeBorder(
            cropped_img, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        return Image.fromarray(squared_img)

# Dynamic Transform Composition
eval_transforms = T.Compose([
    T.Lambda(lambda img: img.convert('RGB')),
    StrictBrainMaskAndCrop(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

### ==========================================
### 3. MODEL LOADING (Cached for Speed with Separation of Status Logs)
### ==========================================
@st.cache_resource
def load_models():
    # 1. Load ViT-Base and apply custom bottleneck head surgery
    vit = models.vit_b_16(weights=None)
    vit.heads.head = nn.Sequential(
        nn.Linear(vit.heads.head.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 4)
    )
    
    # Robust search paths for ViT weights
    vit_loaded = False
    vit_log_status = None
    vit_paths = [
        'models/vit_base_best_weights_v1.pth',
        'vit_base_best_weights_v1.pth',
        'Brain_MRI_Triage_System/models/vit_base_best_weights_v1.pth',
        '../models/vit_base_best_weights_v1.pth'
    ]
    for path in vit_paths:
        try:
            vit.load_state_dict(torch.load(path, map_location=DEVICE))
            vit_loaded = True
            vit_log_status = ("success", f"Loaded ViT weights from: {path}")
            break
        except Exception:
            continue
            
    if not vit_loaded:
        vit_log_status = ("error", "ViT weights not found! Running with random weights.\n💡 Place 'vit_base_best_weights_v1.pth' in a 'models/' folder.")
        
    vit.to(DEVICE).eval()

    # 2. Load Swin Transformer-Base and apply custom bottleneck head surgery
    swin = models.swin_b(weights=None)
    swin.head = nn.Sequential(
        nn.Linear(swin.head.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 4)
    )
    
    # Robust search paths for Swin weights (checking both swin_best_weights_v1.pth and swin_base_best_weights_v1.pth)
    swin_loaded = False
    swin_log_status = None
    swin_paths = [
        'models/swin_best_weights_v1.pth',
        'swin_best_weights_v1.pth',
        'Brain_MRI_Triage_System/models/swin_best_weights_v1.pth',
        'models/swin_base_best_weights_v1.pth',
        'swin_base_best_weights_v1.pth',
        'Brain_MRI_Triage_System/models/swin_base_best_weights_v1.pth',
        '../models/swin_best_weights_v1.pth'
    ]
    for path in swin_paths:
        try:
            swin.load_state_dict(torch.load(path, map_location=DEVICE))
            swin_loaded = True
            swin_log_status = ("success", f"Loaded Swin weights from: {path}")
            break
        except Exception:
            continue
            
    if not swin_loaded:
        swin_log_status = ("error", "Swin weights not found! Running with random weights.\n💡 Place 'swin_best_weights_v1.pth' in a 'models/' folder.")
        
    swin.to(DEVICE).eval()
    
    return vit, swin, [vit_log_status, swin_log_status]

# Load active models
vit_model, swin_model, load_logs = load_models()

# Briefly display load status outside the sidebar using non-intrusive toasts
if "logs_shown" not in st.session_state:
    for status, msg in load_logs:
        if status == "success":
            st.toast(msg, icon="✔️")
        else:
            st.toast(msg, icon="⚠️")
    st.session_state.logs_shown = True

### ==========================================
### 4. INFERENCE & XAI FUNCTIONS
### ==========================================
def predict_ensemble(image_tensor, w_vit=0.6, w_swin=0.4):
    """Executes the Blended Soft-Voting Ensemble (60/40 Default)"""
    with torch.no_grad():
        img_batch = image_tensor.unsqueeze(0).to(DEVICE)
        
        # Forward passes
        vit_logits = vit_model(img_batch)
        swin_logits = swin_model(img_batch)
        
        # Softmax probabilities
        vit_probs = F.softmax(vit_logits, dim=1).cpu().numpy()[0]
        swin_probs = F.softmax(swin_logits, dim=1).cpu().numpy()[0]
        
        # Blend probabilities using soft-voting weights
        blended_probs = w_vit * vit_probs + w_swin * swin_probs
        predicted_idx = np.argmax(blended_probs)
        
        return predicted_idx, blended_probs, vit_probs, swin_probs

def generate_gradcam_masked(image_pil, image_tensor, target_class_idx):
    """Generates a clean, post-hoc anatomically masked Grad-CAM heatmap using ViT-Base"""
    if not HAS_GRAD_CAM:
        return None
        
    def reshape_transform(tensor, height=14, width=14):
        # Vision Transformer patch sequence reshaping transform
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    # Target the last layer normalization block of ViT encoder
    target_layers = [vit_model.encoder.layers.encoder_layer_11.ln_1]
    
    cam = GradCAM(model=vit_model, target_layers=target_layers, reshape_transform=reshape_transform)
    
    # Process scan for model input
    input_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    targets = [ClassifierOutputTarget(target_class_idx)]
    
    # Generate raw heatmaps
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    # Resize and pad the original image using our clinical pipeline
    preprocessed_img = StrictBrainMaskAndCrop()(image_pil)
    preprocessed_resized = preprocessed_img.resize((224, 224))
    img_np = np.float32(preprocessed_resized) / 255.0
    
    # Apply strict Post-Hoc Anatomical Masking (gray threshold of 15) to suppress background voids
    gray_img = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(gray_img, 15, 1, cv2.THRESH_BINARY)
    
    # Element-wise multiplication to strip out background noise
    clean_cam = grayscale_cam * binary_mask
    
    # Overlay the clean heatmap on the remediated scan
    cam_image = show_cam_on_image(img_np, clean_cam, use_rgb=True)
    return cam_image


def plot_horizontal_grouped_bar_chart(classes, vit_probs, swin_probs, blended_probs):
    """Generates a premium horizontal grouped bar chart using Plotly to inherit Streamlit's font and color scheme"""
    import plotly.graph_objects as go
    fig = go.Figure()
    
    # Standalone ViT-Base (Coral Red - Streamlit's default color)
    fig.add_trace(go.Bar(
        y=classes,
        x=[p * 100 for p in vit_probs],
        name='Standalone ViT-Base',
        orientation='h',
        marker=dict(color='#FF4B4B'),
        text=[f" {p*100:.1f}%" if p > 0.05 else "" for p in vit_probs],
        textposition='outside',
        texttemplate='%{text}',
        textfont=dict(size=10, color='var(--text-color)')
    ))
    
    # Standalone Swin (Deep Blue - Streamlit's secondary color)
    fig.add_trace(go.Bar(
        y=classes,
        x=[p * 100 for p in swin_probs],
        name='Standalone Swin-Base',
        orientation='h',
        marker=dict(color='#0068C9'),
        text=[f" {p*100:.1f}%" if p > 0.05 else "" for p in swin_probs],
        textposition='outside',
        texttemplate='%{text}',
        textfont=dict(size=10, color='var(--text-color)')
    ))
    
    # Blended Ensemble (Light Blue - Streamlit's accent color)
    fig.add_trace(go.Bar(
        y=classes,
        x=[p * 100 for p in blended_probs],
        name='Blended Ensemble',
        orientation='h',
        marker=dict(color='#83C9FF'),
        text=[f" {p*100:.1f}%" if p > 0.05 else "" for p in blended_probs],
        textposition='outside',
        texttemplate='%{text}',
        textfont=dict(size=10, color='var(--text-color)', weight='bold')
    ))
    
    fig.update_layout(
        barmode='group',
        xaxis=dict(
            title='Probability (%)', 
            range=[0, 110],  # Give 10% overflow for labels
            gridcolor='rgba(128, 128, 128, 0.15)',
            zeroline=False
        ),
        yaxis=dict(
            autorange="reversed",  # Keeps top class (Glioma) at the top
        ),
        margin=dict(l=10, r=20, t=10, b=10),
        height=320,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

### ==========================================
### 5. STREAMLIT UI/UX DESIGN
### ==========================================
st.title("Multi-Class Brain MRI Triage System")
st.markdown("Developed by **Emmanuel Ovie Mafemi** | MSc Applied Data Science, Teesside University")

# Dynamic White/Black Logo Switcher based on prefers-color-scheme media queries
def get_base64_image(path):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                data = f.read()
            return f"data:image/png;base64,{base64.b64encode(data).decode()}"
        except Exception:
            pass
    return ""

# Search for dark/light mode logos
logo_white_path = "Teesside University White.png"
if not os.path.exists(logo_white_path) and os.path.exists("Brain_MRI_Triage_System/Teesside University White.png"):
    logo_white_path = "Brain_MRI_Triage_System/Teesside University White.png"

logo_black_path = "Teesside University Black.png"
if not os.path.exists(logo_black_path) and os.path.exists("Brain_MRI_Triage_System/Teesside University Black.png"):
    logo_black_path = "Brain_MRI_Triage_System/Teesside University Black.png"

logo_normal_path = "logo.png"
if not os.path.exists(logo_normal_path) and os.path.exists("Brain_MRI_Triage_System/logo.png"):
    logo_normal_path = "Brain_MRI_Triage_System/logo.png"

b64_white = get_base64_image(logo_white_path)
b64_black = get_base64_image(logo_black_path)
b64_normal = get_base64_image(logo_normal_path)


if b64_white:
    st.sidebar.markdown(f"""
        <div class="logo-container">
            <img class="logo-img" src="{b64_white}" />
        </div>
    """, unsafe_allow_html=True)
elif b64_normal:
    st.sidebar.markdown(f"""
        <div class="logo-container">
            <img class="logo-img" src="{b64_normal}" />
        </div>
    """, unsafe_allow_html=True)
else:
    # Stylish fallback title logo that fits the sidebar dimensions perfectly with 0.2cm padding
    st.sidebar.markdown("""
        <div class="logo-container" style="padding-top: 10px; padding-bottom: 10px; border-bottom: 1px solid rgba(128,128,128,0.2); flex-direction: column !important; align-items: center !important;">
            <div style="font-weight: 700; font-size: 1.15rem; color: #4F46E5; letter-spacing: -0.5px;">MIND-Triage AI</div>
            <div style="font-size: 0.72rem; opacity: 0.6; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px;">Clinical Decision Suite</div>
        </div>
    """, unsafe_allow_html=True)


modes = [
    "Urgent Triage (Bulk)", 
    "Clinical Deep Dive (XAI)", 
    "Ensemble Sandbox", 
    "Benchmarking Matrix"
]
selected_mode = st.sidebar.radio("", modes, label_visibility="collapsed")

### PATHWAY 1: URGENT TRIAGE MODE
if selected_mode == "Urgent Triage (Bulk)":
    st.header("⚡ Urgent Triage Mode")
    st.markdown("Upload multiple MRI scans to execute high-speed bulk screening. The **60/40 Champion Ensemble** automatically normalizes the images and outputs a dynamically sorted clinical risk dataframe.")
    
    uploaded_files = st.file_uploader("Upload Scans (JPEG / PNG):", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])
    
    if uploaded_files:
        # Generate an absolute key for the uploaded file collection to track changes
        file_signature = "".join([f"{f.name}_{f.size}" for f in uploaded_files])
        if "triage_signature" not in st.session_state or st.session_state.triage_signature != file_signature:
            st.session_state.triage_results = None
            st.session_state.triage_signature = file_signature

        # Show manual confirmation run button to allow complete file selection first
        run_triage = st.button("⚡ Run Priority Triage Analysis", type="primary")

        # Execute analysis ONLY when confirmed by user
        if run_triage or st.session_state.triage_results is not None:
            if st.session_state.triage_results is None:
                # High-fidelity visual spinner feedback during background execution
                with st.spinner("Processing MRI scans: applying Otsu thresholding, removing skull structures, and executing ensemble predictions..."):
                    start_time = time.time()
                    results = []
                    for file in uploaded_files:
                        img = Image.open(file)
                        img_tensor = eval_transforms(img)
                        
                        # Predict with 60/40 weighted soft-voting
                        pred_idx, probs, _, _ = predict_ensemble(img_tensor, w_vit=0.6, w_swin=0.4)
                        pred_class = CLASSES[pred_idx]
                        confidence = probs[pred_idx] * 100
                        
                        # Formulate clinical risk categories based on pathology severity
                        if pred_class == 'Glioma':
                            risk_score = 1
                            risk_level = '🔴 High Risk (Glioma)'
                        elif pred_class == 'Meningioma':
                            risk_score = 2
                            risk_level = '🟡 Moderate Risk (Meningioma)'
                        elif pred_class == 'Pituitary':
                            risk_score = 3
                            risk_level = '🔵 Low Risk (Pituitary)'
                        else:
                            risk_score = 4
                            risk_level = '🟢 Clear (No Tumour)'
                            
                        results.append({
                            "Filename": file.name,
                            "Clinical Status": risk_level,
                            "Predicted Class": pred_class,
                            "Confidence": f"{confidence:.2f}%",
                            "Risk Sort Key": risk_score
                        })
                    end_time = time.time()
                    st.session_state.triage_results = results
                    st.session_state.triage_latency = (end_time - start_time) * 1000

            # Create Pandas DataFrame and sort dynamically by risk severity (Gliomas first)
            df = pd.DataFrame(st.session_state.triage_results)
            df_sorted = df.sort_values(by="Risk Sort Key").drop(columns=["Risk Sort Key"]).reset_index(drop=True)
            df_sorted.index = df_sorted.index + 1
            
            st.subheader("📋 Priority Patient Queue")
            df_styled = df_sorted.style.highlight_max(subset=["Confidence"], color="#ffcdd2")
            st.dataframe(df_styled, width="stretch")
            scan_word = "scan" if len(uploaded_files) == 1 else "scans"
            st.success(f"Triaged {len(uploaded_files)} {scan_word} in {st.session_state.triage_latency:.2f} ms on this machine.")
            st.info(f"**Benchmark Reference:** On the NVIDIA Tesla T4 GPU, this exact triage operation takes an average of **{len(uploaded_files) * 27.85:.2f} ms** (27.85 ms per scan).")

### PATHWAY 2: CLINICAL DEEP DIVE MODE (XAI)
elif selected_mode == "Clinical Deep Dive (XAI)":
    st.header("🔬 Clinical Deep Dive Mode")
    st.markdown("Execute detailed visual auditing on a single brain scan. The **ViT-Base** architecture generates an anatomically masked Grad-CAM attention heatmap to identify the precise boundaries of the pathology.")
    
    uploaded_file = st.file_uploader("Upload Scan:", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        file_signature = f"{uploaded_file.name}_{uploaded_file.size}"
        if "dive_signature" not in st.session_state or st.session_state.dive_signature != file_signature:
            st.session_state.dive_results = None
            st.session_state.dive_signature = file_signature

        # Run button for deep visual audits
        run_dive = st.button("🔬 Execute Clinical Deep Dive Audit", type="primary")

        if run_dive or st.session_state.dive_results is not None:
            if st.session_state.dive_results is None:
                with st.spinner("Extracting anatomical coordinates, mapping attention heads, and generating post-hoc masked Grad-CAM overlays..."):
                    img = Image.open(uploaded_file)
                    img_tensor = eval_transforms(img)
                    
                    # Predict using the standalone ViT model path (w_vit=1.0)
                    pred_idx, blended_probs, vit_probs, swin_probs = predict_ensemble(img_tensor, w_vit=1.0, w_swin=0.0)
                    
                    # Crop image for side-by-side display
                    clean_img = StrictBrainMaskAndCrop()(img)
                    
                    # Generate Grad-CAM heatmaps if available
                    cam_overlay = None
                    if HAS_GRAD_CAM:
                        cam_overlay = generate_gradcam_masked(img, img_tensor, pred_idx)
                        
                    st.session_state.dive_results = {
                        "pred_idx": pred_idx,
                        "vit_probs": vit_probs,
                        "clean_img": clean_img,
                        "cam_overlay": cam_overlay,
                        "original_img": img
                    }

            # Retrieve results from state
            res = st.session_state.dive_results
            col1, col2, col3 = st.columns(3)
            
            # Use custom small headers with strict no-wrap style to keep alignments perfect
            with col1:
                st.markdown("<h4 style='font-size: 1.1rem; margin-top: 0px; margin-bottom: 0.5rem; white-space: nowrap;'>📷 Original Scan</h4>", unsafe_allow_html=True)
                st.image(res["original_img"], width="stretch")
                
            with col2:
                st.markdown("<h4 style='font-size: 1.1rem; margin-top: 0px; margin-bottom: 0.5rem; white-space: nowrap;'>🛡️ Isolated Brain Tissue</h4>", unsafe_allow_html=True)
                st.image(res["clean_img"], width="stretch")
                
            with col3:
                st.markdown("<h4 style='font-size: 1.1rem; margin-top: 0px; margin-bottom: 0.5rem; white-space: nowrap;'>🔥 Masked Grad-CAM</h4>", unsafe_allow_html=True)
                if res["cam_overlay"] is not None:
                    st.image(res["cam_overlay"], width="stretch")
                else:
                    st.info("PyTorch Grad-CAM library is not loaded. Displaying un-audited results.")
                    
            # Display probabilistic breakdown
            st.subheader("📊 Diagnostic Probabilities")
            prob_df = pd.DataFrame({
                "Classification Pathology": CLASSES,
                "ViT Probability": [f"{p*100:.2f}%" for p in res["vit_probs"]]
            })
            st.table(prob_df)

### PATHWAY 3: ENSEMBLE SANDBOX MODE
elif selected_mode == "Ensemble Sandbox":
    st.header("🎛️ Ensemble Sandbox")
    st.markdown("Interactively adjust the mathematical weights of the probabilistic soft-voting ensemble to see how fusing the global spatial reasoning of ViT and local hierarchy of Swin shifts classification confidence.")
    
    # Sliding controls for clinical simulation
    w_vit = st.slider("ViT-Base Voting Weight (wa):", 0.0, 1.0, 0.60, step=0.05)
    w_swin = 1.0 - w_vit
    st.info(f"Loaded Configuration: **{w_vit*100:.0f}% ViT-Base + {w_swin*100:.0f}% Swin Transformer**")
    
    uploaded_file = st.file_uploader("Upload Scan:", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        file_signature = f"{uploaded_file.name}_{uploaded_file.size}_{w_vit:.2f}"
        if "sandbox_signature" not in st.session_state or st.session_state.sandbox_signature != file_signature:
            st.session_state.sandbox_results = None
            st.session_state.sandbox_signature = file_signature

        # Run button for testing custom ensembling
        run_sandbox = st.button("🎛️ Compute Blended Probability Matrix", type="primary")

        if run_sandbox or st.session_state.sandbox_results is not None:
            if st.session_state.sandbox_results is None:
                with st.spinner("Executing forward passes and computing blended soft-voting metrics..."):
                    img = Image.open(uploaded_file)
                    img_tensor = eval_transforms(img)
                    
                    # Evaluate using custom weights
                    pred_idx, blended_probs, vit_probs, swin_probs = predict_ensemble(img_tensor, w_vit=w_vit, w_swin=w_swin)
                    
                    st.session_state.sandbox_results = {
                        "pred_idx": pred_idx,
                        "blended_probs": blended_probs,
                        "vit_probs": vit_probs,
                        "swin_probs": swin_probs
                    }

            res = st.session_state.sandbox_results
            st.subheader(f"Ensemble Prediction: **{CLASSES[res['pred_idx']]}** ({res['blended_probs'][res['pred_idx']]*100:.2f}% Confidence)")
            
            # Render custom horizontal grouped bar chart
            fig = plot_horizontal_grouped_bar_chart(CLASSES, res["vit_probs"], res["swin_probs"], res["blended_probs"])
            st.plotly_chart(fig, width="stretch")
            
            # Clean, scientifically accurate Blended Probability Matrix
            st.markdown("### 📊 Blended Probability Matrix")
            matrix_df = pd.DataFrame({
                "Standalone ViT-Base": [f"{p*100:.2f}%" for p in res["vit_probs"]],
                "Standalone Swin-Base": [f"{p*100:.2f}%" for p in res["swin_probs"]],
                f"Blended Ensemble ({w_vit*100:.0f}/{w_swin*100:.0f})": [f"{p*100:.2f}%" for p in res["blended_probs"]]
            }, index=CLASSES)
            st.table(matrix_df)

### PATHWAY 4: BENCHMARKING MATRIX
elif selected_mode == "Benchmarking Matrix":
    st.header("📊 Clinical Metrics & Benchmarking Matrix")
    st.markdown("Final performance metrics across the strictly locked, unseen testing set of 1,600 MRI scans (400 per class) with exact hardware parameter benchmarks.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 style='font-size: 1.1rem; font-weight: 700; margin-top: 0px; margin-bottom: 0.8rem; white-space: nowrap;'>🏆 Pairwise Ensemble Leaderboard</h3>", unsafe_allow_html=True)
        leaderboard_data = {
            "Rank": [1, 2, 3, 4, 5],
            "Ensemble Pairing": [
                "ViT-Base + Swin Transformer (Champion)",
                "ViT-Base + ConvNeXt-Base",
                "ResNet-50 + ViT-Base",
                "EfficientNet-B0 + ViT-Base",
                "Swin Transformer + ConvNeXt-Base"
            ],
            "Overall Accuracy": ["93.31%", "92.81%", "92.25%", "91.94%", "90.88%"],
            "Macro F1": ["93.24%", "92.69%", "92.12%", "91.81%", "90.73%"],
            "Glioma Recall": ["80.50%", "78.75%", "77.25%", "75.75%", "76.50%"],
            "Healthy Clearance": ["99.75%", "100.00%", "100.00%", "100.00%", "99.25%"]
        }
        st.table(pd.DataFrame(leaderboard_data))
        st.markdown("**Clinical Justification for 60/40 Weighting:** Though the 50/50 unweighted blended ensemble peaked at the same 93.31% global accuracy, the 60% ViT-Base majority was selected for production because it caught the absolute maximum number of aggressive gliomas (324 out of 400) while restoring the standalone ViT's flawless healthy clearance rate (exactly zero false-positive alerts).")

    with col2:
        st.markdown("<h3 style='font-size: 1.1rem; font-weight: 700; margin-top: 0px; margin-bottom: 0.8rem; white-space: nowrap;'>⚡ Hardware & Latency Benchmark</h3>", unsafe_allow_html=True)
        latency_data = {
            "Architecture": [
                "EfficientNet-B0",
                "ResNet-50",
                "ConvNeXt-Base",
                "Swin Transformer-Base",
                "ViT-Base",
                "60/40 Ensemble (ViT+Swin)"
            ],
            "Parameters": ["4,665,472", "24,559,172", "88,093,316", "87,270,076", "86,194,436", "173,464,512"],
            "FLOPs (Ops)": ["770M", "8.17G", "30.70G", "30.86G", "22.54G", "53.40G"],
            "Average Latency (ms)": ["2.45 ms", "4.85 ms", "8.12 ms", "12.18 ms", "15.40 ms", "27.85 ms"]
        }
        st.table(pd.DataFrame(latency_data))
        st.markdown("Operating at **27.85 milliseconds per scan**, the 60/40 weighted soft-voting ensemble triages a massive hospital queue of 100 patient brain scans in only 2.78 seconds, effectively mitigating the clinical radiologist throughput bottleneck.")
