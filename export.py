import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO


def export_yolo(config_path="config.yaml", weights_override=None, format_override=None):
    """
    Exports a trained YOLOv11 model to specified deployment formats (ONNX, TorchScript, etc.).
    """
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    export_cfg = config.get("export", {})

    weights = weights_override or export_cfg.get("weights", "runs/detect/yolov11_custom_train/weights/best.pt")
    export_format = format_override or export_cfg.get("format", "onnx")

    if not Path(weights).is_file():
        raise FileNotFoundError(
            f"Fine-tuned weights not found: {weights}. Train the model first or pass --weights."
        )

    print("=" * 60)
    print("📦 EXPORTING YOLOV11 MODEL")
    print("=" * 60)
    print(f"Weights Source : {weights}")
    print(f"Target Format  : {export_format.upper()}")
    print(f"Dynamic Shape  : {export_cfg.get('dynamic', False)}")
    print(f"FP16 Half Prec : {export_cfg.get('half', False)}")
    print("=" * 60)

    model = YOLO(weights)
    exported_path = model.export(
        format=export_format,
        dynamic=export_cfg.get("dynamic", False),
        half=export_cfg.get("half", False)
    )

    print("\n" + "=" * 60)
    print("✅ MODEL EXPORT COMPLETED")
    print("=" * 60)
    print(f"Exported File Path: {exported_path}")
    print("=" * 60 + "\n")

    return exported_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv11 model.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    parser.add_argument("--weights", type=str, default=None, help="Weights file (.pt)")
    parser.add_argument("--format", type=str, default=None, help="Target format (onnx, torchscript, openvino, engine)")
    args = parser.parse_args()

    export_yolo(args.config, args.weights, args.format)
