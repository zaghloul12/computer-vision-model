import io
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


@st.cache_resource
def load_model(weights_path: str):
    return YOLO(weights_path)


def main():
    st.set_page_config(page_title="YOLOv8 Segmentation Demo", layout="wide")
    st.title("YOLOv8 Segmentation Demo")

    with st.sidebar:
        st.header("Model Settings")
        weights_path = st.text_input("Weights path", "runs/segment/custom/weights/best.pt")
        conf = st.slider("Confidence", 0.05, 0.9, 0.25, 0.05)
        imgsz = st.selectbox("Image size", [320, 512, 640, 768, 1024], index=2)
        device = st.text_input("Device", "")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    if not uploaded:
        st.info("Upload an image to run inference.")
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input", use_container_width=True)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        img_path = tmp.name

    model = load_model(weights_path)
    results = model.predict(
        source=img_path,
        conf=conf,
        imgsz=imgsz,
        device=device if device.strip() else None,
        save=False,
    )

    result = results[0]
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, caption="Prediction", use_container_width=True)

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
