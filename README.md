# YOLOv8 Segmentation (COCO -> Local)

This project fine-tunes YOLOv8-seg on a local COCO-format dataset and provides inference/export scripts.

## 1) Install

```bash
pip install -r requirements.txt
```

## 2) Prepare dataset

Make sure your dataset has COCO JSON annotations and images. Example:

```
C:\Users\Dell\Downloads\My First Project.coco\
  annotations.json
  images\
```

Run conversion to YOLOv8 segmentation labels and create `data.yaml`:

```bash
python prepare_data.py --coco-json "C:\Users\Dell\Downloads\My First Project.coco\annotations.json" --images-dir "C:\Users\Dell\Downloads\My First Project.coco\images" --out-dir "C:\Users\Dell\Downloads\My First Project.coco"
```

This writes:
- `labels/` with YOLOv8 segmentation labels
- `train.txt` / `val.txt`
- `data.yaml`

## 3) Train

```bash
python train.py --data "C:\Users\Dell\Downloads\My First Project.coco\data.yaml" --epochs 50 --batch 8 --imgsz 640
```

Trained weights appear in `runs/segment/custom/weights/`.

## 4) Test (inference)

```bash
python infer.py --weights "runs/segment/custom/weights/best.pt" --source "C:\Users\Dell\Downloads\My First Project.coco\images"
```

Results are saved under `runs/predict/`.

## 5) Export (deploy)

```bash
python export_model.py --weights "runs/segment/custom/weights/best.pt" --format onnx
```

The exported model is saved next to the weights.

## 6) Streamlit demo app

```bash
streamlit run streamlit_app.py
```

In the sidebar, set the path to your trained weights (default: `runs/segment/custom/weights/best.pt`).
