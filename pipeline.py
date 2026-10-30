import os
import yaml
from pathlib import Path
from dataset_utils import prepare_dataset_yaml
from train import train_yolo
from val import validate_yolo
from predict import run_prediction
from export import export_yolo


class YOLOPipeline:
    """
    Complete Object Detection Pipeline using YOLOv11.
    Handles dataset verification, fine-tuning, evaluation, prediction, and model export.
    """

    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.dataset_yaml = None
        self.best_weights_path = None

    def prepare_data(self):
        """Verifies train/valid/test datasets and builds dataset YAML."""
        print("\n--- STEP 1: DATASET PREPARATION & VALIDATION ---")
        self.dataset_yaml = prepare_dataset_yaml(self.config, self.config["dataset"].get("data_yaml", "dataset.yaml"))
        return self.dataset_yaml

    def train(self):
        """Fine-tunes YOLOv11 on the custom dataset."""
        print("\n--- STEP 2: MODEL FINE-TUNING ---")
        self.best_weights_path = train_yolo(self.config_path)
        return self.best_weights_path

    def evaluate(self, split="val", weights_path=None):
        """Evaluates the model on validation or test set."""
        print(f"\n--- STEP 3: MODEL EVALUATION ({split.upper()}) ---")
        weights = weights_path or self.best_weights_path
        return validate_yolo(self.config_path, weights_path=weights, split=split)

    def predict(self, source=None, weights_path=None):
        """Runs object detection on images/videos/folder."""
        print("\n--- STEP 4: INFERENCE / PREDICTION ---")
        weights = weights_path or self.best_weights_path
        return run_prediction(self.config_path, source_override=source, weights_override=weights)

    def export(self, export_format="onnx", weights_path=None):
        """Exports model to target format (e.g. ONNX)."""
        print("\n--- STEP 5: MODEL EXPORT ---")
        weights = weights_path or self.best_weights_path
        return export_yolo(self.config_path, weights_override=weights, format_override=export_format)

    def run_full_pipeline(self):
        """Runs dataset preparation, fine-tuning, validation, prediction, and export sequentially."""
        print("=" * 70)
        print("🌟 STARTING COMPLETE YOLOV11 PIPELINE")
        print("=" * 70)
        
        self.prepare_data()
        self.train()
        self.evaluate(split="val")
        self.predict()
        self.export()

        print("=" * 70)
        print("✨ YOLOV11 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)


if __name__ == "__main__":
    pipeline = YOLOPipeline("config.yaml")
    pipeline.prepare_data()
