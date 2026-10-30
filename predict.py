import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO


def run_prediction(config_path="config.yaml", source_override=None, weights_override=None):
    """
    Runs YOLOv11 inference on input images, directories, or videos.
    """
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    predict_cfg = config.get("predict", {})

    weights = weights_override or predict_cfg.get("weights", "runs/detect/yolov11_custom_train/weights/best.pt")
    source = source_override or predict_cfg.get("source", "Counting Screws.v8i.yolov11/test/images")

    if not Path(weights).is_file():
        raise FileNotFoundError(
            f"Fine-tuned weights not found: {weights}. Train the model first or pass --weights."
        )

    if not Path(source).exists():
        raise FileNotFoundError(f"Inference source path not found: {source}")

    print("=" * 60)
    print("🔍 RUNNING YOLOV11 OBJECT DETECTION INFERENCE")
    print("=" * 60)
    print(f"Model Weights : {weights}")
    print(f"Input Source  : {source}")
    print(f"Confidence    : {predict_cfg.get('conf', 0.25)}")
    print(f"IOU Threshold : {predict_cfg.get('iou', 0.45)}")
    print("=" * 60)

    model = YOLO(weights)
    results = model.predict(
        source=source,
        conf=predict_cfg.get("conf", 0.25),
        iou=predict_cfg.get("iou", 0.45),
        save=predict_cfg.get("save", True),
        save_txt=predict_cfg.get("save_txt", False),
        project=predict_cfg.get("project", "runs/detect"),
        name=predict_cfg.get("name", "yolov11_predictions"),
        exist_ok=True
    )

    print("\n" + "=" * 60)
    print("🎯 INFERENCE SUMMARY")
    print("=" * 60)
    
    total_detections = 0
    for idx, r in enumerate(results):
        num_boxes = len(r.boxes)
        total_detections += num_boxes
        img_name = Path(r.path).name if hasattr(r, 'path') else f"Image {idx+1}"
        
        # Count classes per image
        class_counts = {}
        if num_boxes > 0:
            for cls_id in r.boxes.cls.tolist():
                cls_name = model.names[int(cls_id)]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        
        counts_str = ", ".join([f"{count} {name}(s)" for name, count in class_counts.items()]) if class_counts else "No objects detected"
        print(f"[{idx+1}/{len(results)}] {img_name} -> {counts_str}")

    save_dir = results[0].save_dir if len(results) > 0 and hasattr(results[0], 'save_dir') else "runs/detect/yolov11_predictions"
    print("-" * 60)
    print(f"Total Objects Identified : {total_detections}")
    print(f"Annotated Output Saved At: {save_dir}")
    print("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv11 inference on test images.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--source", type=str, default=None, help="Image path or directory for inference")
    parser.add_argument("--weights", type=str, default=None, help="Weights file path (.pt)")
    args = parser.parse_args()

    run_prediction(args.config, args.source, args.weights)
