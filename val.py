import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
from dataset_utils import prepare_dataset_yaml


def validate_yolo(config_path="config.yaml", weights_path=None, split="val"):
    """
    Evaluates a trained YOLOv11 model on validation or test split.
    """
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Ensure dataset yaml exists
    dataset_yaml_path = prepare_dataset_yaml(config, output_yaml_path=config["dataset"].get("data_yaml", "dataset.yaml"))

    val_cfg = config.get("val", {})
    if not weights_path:
        weights_path = config.get("predict", {}).get("weights", "runs/detect/yolov11_custom_train/weights/best.pt")
    if not Path(weights_path).is_file():
        raise FileNotFoundError(
            f"Fine-tuned weights not found: {weights_path}. Train the model first or pass --weights."
        )

    print("=" * 60)
    print("📈 EVALUATING YOLOV11 MODEL")
    print("=" * 60)
    print(f"Weights      : {weights_path}")
    print(f"Dataset YAML : {dataset_yaml_path}")
    print(f"Split        : {split}")
    print(f"Image Size   : {val_cfg.get('imgsz', 640)}")
    print("=" * 60)

    model = YOLO(weights_path)
    metrics = model.val(
        data=dataset_yaml_path,
        split=split,
        imgsz=val_cfg.get("imgsz", 640),
        batch=val_cfg.get("batch_size", 16),
        conf=val_cfg.get("conf", 0.25),
        iou=val_cfg.get("iou", 0.6)
    )

    print("\n" + "=" * 60)
    print("📊 EVALUATION METRICS SUMMARY")
    print("=" * 60)
    print(f"Precision (P)    : {metrics.box.mp:.4f}")
    print(f"Recall (R)       : {metrics.box.mr:.4f}")
    print(f"mAP @ 0.50       : {metrics.box.map50:.4f}")
    print(f"mAP @ 0.50:0.95  : {metrics.box.map:.4f}")
    print("=" * 60 + "\n")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv11 model.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights (.pt)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate")
    args = parser.parse_args()

    validate_yolo(args.config, args.weights, args.split)
