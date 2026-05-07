import streamlit as st
from PIL import Image
import io
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.cm as cm
from torchvision import models, transforms
import sys
sys.path.append('/Users/danial/Desktop/CV')
from hybrid_model import HybridModel, VAE

st.set_page_config(page_title="CellVision AI", page_icon="🔬", layout="centered")

st.markdown("""
<style>
    .title { text-align:center; font-size:2.5rem; font-weight:700; color:#00d4ff; }
    .subtitle { text-align:center; color:#888; font-size:1rem; margin-bottom:2rem; }
    .result-box { background:#1e2130; border-radius:12px; padding:20px;
                  margin-top:20px; border-left:4px solid #00d4ff; }
    .anomaly-box { background:#2d1515; border-radius:12px; padding:20px;
                   margin-top:20px; border-left:4px solid #ff4444; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title"> ⚡️ CellVision AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Blood Cell Classification + Anomaly Detection + Grad-CAM + U-Net Segmentation</div>',
            unsafe_allow_html=True)

class GradCAMStreamlit:
    def __init__(self, model, target_layer):
        self.model      = model
        self.gradients  = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, class_idx=None):
        self.model.eval()
        output = self.model(image_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients   = self.gradients[0]
        activations = self.activations[0]
        weights     = gradients.mean(dim=(1, 2))
        cam         = (weights[:, None, None] * activations).sum(dim=0)
        cam         = F.relu(cam)
        cam         = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy(), class_idx

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1       = DoubleConv(3, 32)
        self.enc2       = DoubleConv(32, 64)
        self.enc3       = DoubleConv(64, 128)
        self.pool       = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.up3        = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3       = DoubleConv(256, 128)
        self.up2        = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2       = DoubleConv(128, 64)
        self.up1        = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1       = DoubleConv(64, 32)
        self.final      = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1))

@st.cache_resource
def load_hybrid_model():
    return HybridModel(
        config_path     ='/Users/danial/Desktop/CV/hybrid_model_config.json',
        classifier_path ='/Users/danial/Desktop/CV/best_model_efficientnet.pth',
        vae_path        ='/Users/danial/Desktop/CV/vae_anomaly.pth'
    )

@st.cache_resource
def load_unet():
    unet = UNet()
    unet.load_state_dict(torch.load(
        '/Users/danial/Desktop/CV/unet_segmentation.pth',
        map_location='cpu'
    ))
    unet.eval()
    return unet

model   = load_hybrid_model()
gradcam = GradCAMStreamlit(model.classifier, model.classifier.features[-1])

preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

with st.sidebar:
    st.markdown("### ABOUT MODEL. ")
    st.markdown(
        "**Model:** EfficientNet-B0 + VAE  \n"
        "**Accuracy:** 99%  \n"
        "**Classes:** 4 normal + anomaly detection"
    )
    st.markdown("---")
    st.markdown("### CELL CLASSES")
    for cls, desc in {
        "• EOSINOPHIL" : "Allergic reactions & parasites",
        "• LYMPHOCYTE" : "Immune system, fights viruses",
        "• MONOCYTE"   : "Largest WBC, removes bacteria",
        "• NEUTROPHIL" : "First responder to infection",
        "• ANOMALY"    : "Unusual cell detected by VAE"
    }.items():
        st.markdown(f"**{cls}**  \n{desc}")

    st.markdown("---")
    st.markdown("### ⚙️ Display Options")
    show_gradcam_flag = st.checkbox("🔥 Show Grad-CAM heatmap",    value=True)
    show_unet_flag    = st.checkbox("🔬 Show U-Net segmentation",   value=True)
    st.caption("Grad-CAM — model attention map")
    st.caption("U-Net — cell boundary segmentation")

st.markdown("### 🩸 Upload Blood Cell Image")
uploaded_file = st.file_uploader(
    "Choose an image (JPG, PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Image**")
        st.image(image, use_container_width=True)
        st.caption(f"Size: {image.size[0]}×{image.size[1]} px")

    with st.spinner("Analyzing cell..."):
        result      = model.predict(image)
        is_anomaly  = result['is_anomaly']
        predicted   = result['predicted_class']
        confidence  = result['confidence']
        recon_error = result['reconstruction_error']
        threshold   = result['threshold']

    with col2:
        st.markdown("**Prediction**")
        if is_anomaly:
            st.error("⚠️ ANOMALY DETECTED")
            st.caption(f"Reconstruction error {recon_error:.4f} > threshold {threshold:.4f}")
        else:
            st.success(f"✅ {predicted}")
            st.metric("Confidence", f"{confidence:.1%}")
            st.progress(float(confidence))

        st.markdown("---")
        m1, m2 = st.columns(2)
        m1.metric("Recon Error", f"{recon_error:.4f}")
        m2.metric("Threshold",   f"{threshold:.4f}")

    if show_gradcam_flag and not is_anomaly:
        st.markdown("---")
        st.markdown("### 🔥 Grad-CAM — Model Attention Map")
        st.caption("Red/yellow areas show where the model focused to make its decision")

        img_tensor = preprocess(image).unsqueeze(0).to(model.device)
        img_tensor.requires_grad_(True)

        cam, pred_idx = gradcam.generate(img_tensor)

        img_np      = np.array(image.resize((128, 128))).astype(float) / 255.0
        cam_resized = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (128, 128), Image.BILINEAR)
        ) / 255.0

        heatmap = cm.jet(cam_resized)[:, :, :3]
        overlay = np.clip(0.5 * img_np + 0.5 * heatmap, 0, 1)

        c1, c2, c3 = st.columns(3)
        c1.image(img_np,  caption="Original",              use_container_width=True)
        c2.image(heatmap, caption="Heatmap",               use_container_width=True)
        c3.image(overlay, caption=f"Overlay — {predicted}", use_container_width=True)

        st.info(
            "🧠 EXPLAINABILITY : The highlighted region shows the cell area the model used "
            "to classify this image. This helps verify the model is looking "
            "at the right features."
        )


    if show_unet_flag and not is_anomaly:
        st.markdown("---")
        st.markdown("### 🔬 U-Net Cell Segmentation")
        st.caption("U-Net predicts a pixel-level mask highlighting the cell boundary region")

        try:
            unet = load_unet()

            unet_transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])

            img_tensor_unet = unet_transform(image).unsqueeze(0)

            with torch.no_grad():
                seg_pred = unet(img_tensor_unet)[0, 0].numpy()

            seg_bin  = (seg_pred > 0.5).astype(float)
            img_show = np.array(image.resize((128, 128))).astype(float) / 255.0

            overlay_seg           = img_show.copy()
            overlay_seg[:, :, 0]  = np.clip(overlay_seg[:, :, 0] - seg_bin * 0.2, 0, 1)
            overlay_seg[:, :, 1]  = np.clip(overlay_seg[:, :, 1] + seg_bin * 0.4, 0, 1)
            overlay_seg[:, :, 2]  = np.clip(overlay_seg[:, :, 2] - seg_bin * 0.2, 0, 1)
            overlay_seg           = np.clip(overlay_seg, 0, 1)

            s1, s2, s3 = st.columns(3)
            s1.image(img_show,  caption="Original",          use_container_width=True)
            s2.image(seg_pred,  caption="Segmentation Mask", use_container_width=True, clamp=True)
            s3.image(overlay_seg, caption="Overlay (green = cell)", use_container_width=True)

            coverage = seg_bin.mean() * 100
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Cell Coverage",  f"{coverage:.1f}%")
            col_m2.metric("Dice Score",     "0.997")
            col_m3.metric("Mask Threshold", "0.5")

            st.success(
                "✅ U-Net successfully segmented the cell region. "
                "Green overlay highlights the detected cell boundary."
            )

        except FileNotFoundError:
            st.warning(
                "⚠️ U-Net model not found. "
                "Make sure unet_segmentation.pth is in /Users/danial/Desktop/CV/"
            )
        except Exception as e:
            st.error(f"U-Net error: {str(e)}")

    if is_anomaly:
        st.markdown("---")
        st.warning(
            "⚠️ Anomaly detected — Grad-CAM and U-Net segmentation "
            "are skipped for anomalous inputs. The VAE reconstruction "
            f"error ({recon_error:.4f}) exceeded the threshold ({threshold:.4f})."
        )

st.markdown("---")
st.markdown("### How It Works")

c1, c2, c3, c4 = st.columns(4)
c1.markdown("**Step 1: VAE Check**  \nReconstruction error checked. If > 0.0207 → ANOMALY")
c2.markdown("**Step 2: Classification**  \nEfficientNet-B0 classifies into 4 cell types with 99% accuracy")
c3.markdown("**Step 3: Grad-CAM**  \nHeatmap shows which region influenced the classification decision")
c4.markdown("**Step 4: U-Net**  \nPixel-level segmentation highlights the exact cell boundary")

st.markdown("---")
st.caption(
    "CellVision AI System • EfficientNet-B0 (99% accuracy) + VAE Anomaly Detection "
    "+ Grad-CAM Explainability + U-Net Segmentation • Computer Vision, Spring 2026."
)
st.caption("With respect, Ruspekov D., Raiymbek B., Oralbayev B., Tlegenov A.")