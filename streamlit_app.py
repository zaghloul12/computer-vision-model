import io
import os
import tempfile

import streamlit as st
from PIL import Image
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel


@st.cache_resource
def load_model(weights_path: str):
    torch.serialization.add_safe_globals([SegmentationModel])
    return YOLO(weights_path)


def main():
    st.set_page_config(page_title="YOLOv8 Segmentation Demo", layout="wide")
    st.title("YOLOv8 Segmentation Demo")

    with st.sidebar:
        st.header("Model Settings")
        weights_path = st.text_input("Weights path", "runs/segment/custom/weights/best.pt")
        uploaded_weights = st.file_uploader("Or upload weights (.pt)", type=["pt"])
        conf = st.slider("Confidence", 0.05, 0.9, 0.25, 0.05)
        imgsz = st.selectbox("Image size", [320, 512, 640, 768, 1024], index=2)
        device = st.text_input("Device", "")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    if not uploaded:
        st.info("Upload an image to run inference.")
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input", width="stretch")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        img_path = tmp.name

    model_path = weights_path
    if uploaded_weights is not None:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_w:
            tmp_w.write(uploaded_weights.read())
            model_path = tmp_w.name
    elif not os.path.exists(weights_path):
        st.warning("Weights not found. Using pretrained yolov8n-seg.pt.")
        model_path = "yolov8n-seg.pt"

    model = load_model(model_path)
    results = model.predict(
        source=img_path,
        conf=conf,
        imgsz=imgsz,
        device=device if device.strip() else None,
        save=False,
    )

    result = results[0]
    annotated_bgr = result.plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]
    st.image(annotated_rgb, caption="Prediction", width="stretch")

    buffer = io.BytesIO()
    Image.fromarray(annotated_rgb).save(buffer, format="PNG")
    st.download_button(
        "Download prediction",
        data=buffer.getvalue(),
        file_name="prediction.png",
        mime="image/png",
    )


if __name__ == "__main__":
    main()
