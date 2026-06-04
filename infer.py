import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to trained .pt file")
    parser.add_argument("--source", required=True, help="Image, video, or folder path")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save-dir", default="runs/predict")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        conf=args.conf,
        save=True,
        save_txt=True,
        save_conf=True,
        project=args.save_dir,
        show=args.show,
    )


if __name__ == "__main__":
    main()
