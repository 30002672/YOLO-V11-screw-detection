import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
from dataset_utils import prepare_dataset_yaml


def train_yolo(config_path="config.yaml"):
    """
    Fine-tunes a YOLOv11 model based on parameters specified in config.yaml.
    """
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Prepare and validate dataset configuration file
    dataset_yaml_path = prepare_dataset_yaml(config, output_yaml_path=config["dataset"].get("data_yaml", "dataset.yaml"))

    model_type = config.get("model", {}).get("type", "yolo11n.pt")
    model_path = Path(model_type)
    if model_path.suffix == ".pt" and not model_path.is_absolute() and model_path.exists():
        model_type = str(model_path.resolve())
    train_args = config.get("train", {})
    project_path = Path(train_args.get("project", "runs/detect"))
    if not project_path.is_absolute():
        project_path = config_file.parent / project_path
    project_path = project_path.resolve()

    print("=" * 60)
    print(f"🚀 INITIALIZING YOLOV11 TRAINING")
    print("=" * 60)
    print(f"Base Model    : {model_type}")
    print(f"Dataset YAML  : {dataset_yaml_path}")
    print(f"Epochs        : {train_args.get('epochs', 50)}")
    print(f"Batch Size    : {train_args.get('batch_size', 16)}")
    print(f"Image Size    : {train_args.get('imgsz', 640)}")
    print(f"Device        : {train_args.get('device', 'auto')}")
    print(f"Project Output: {project_path}/{train_args.get('name', 'yolov11_custom')}")
    print("=" * 60)

    # Load YOLOv11 pretrained model
    model = YOLO(model_type)

    # Execute training/fine-tuning
    results = model.train(
        data=dataset_yaml_path,
        epochs=train_args.get("epochs", 50),
        batch=train_args.get("batch_size", 16),
        imgsz=train_args.get("imgsz", 640),
        device=train_args.get("device", "auto"),
        workers=train_args.get("workers", 4),
        patience=train_args.get("patience", 20),
        save_period=train_args.get("save_period", -1),
        project=str(project_path),
        name=train_args.get("name", "yolov11_custom_train"),
        exist_ok=train_args.get("exist_ok", True),
        pretrained=train_args.get("pretrained", True),
        optimizer=train_args.get("optimizer", "auto"),
        lr0=train_args.get("lr0", 0.01),
        lrf=train_args.get("lrf", 0.01),
        seed=train_args.get("seed", 42)
    )

    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Training completed without producing best weights: {best_weights}")

    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Best Weights Saved At: {best_weights}")
    print(f"Training Artifacts Dir: {save_dir}")
    print("=" * 60 + "\n")

    return str(best_weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv11 model on custom dataset.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration YAML file")
    args = parser.parse_args()

    train_yolo(args.config)
