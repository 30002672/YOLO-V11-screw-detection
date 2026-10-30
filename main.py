import argparse
from pipeline import YOLOPipeline


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 Fine-Tuning & Object Detection Pipeline CLI")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["prepare", "train", "val", "predict", "export", "full"],
        help="Action to perform (default: train): prepare, train, val, predict, export, full"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--source", type=str, default=None, help="Image/video path or folder for prediction")
    parser.add_argument("--weights", type=str, default=None, help="Weights path (.pt) for val/predict/export")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split for val mode")
    parser.add_argument("--format", type=str, default="onnx", help="Export format (onnx, torchscript, openvino, engine, etc.)")

    args = parser.parse_args()

    pipeline = YOLOPipeline(config_path=args.config)

    if args.mode == "prepare":
        pipeline.prepare_data()

    elif args.mode == "train":
        pipeline.prepare_data()
        pipeline.train()

    elif args.mode == "val":
        pipeline.evaluate(split=args.split, weights_path=args.weights)

    elif args.mode == "predict":
        pipeline.predict(source=args.source, weights_path=args.weights)

    elif args.mode == "export":
        pipeline.export(export_format=args.format, weights_path=args.weights)

    elif args.mode == "full":
        pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
