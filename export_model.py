import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to trained .pt file")
    parser.add_argument("--format", default="onnx", help="onnx, torchscript, openvino, engine")
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.export(format=args.format)


if __name__ == "__main__":
    main()
