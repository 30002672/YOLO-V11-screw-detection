# Fine-tune YOLO11 to count screws

This project fine-tunes the supplied `yolo11n.pt` detector on the screw dataset in
`Counting Screws.v8i.yolov11/`. The trained model detects screws; the prediction
step counts the detections in each image.

## Project parts

1. **Prepare the data** — inspect train/validation/test images and labels, then
   generate `dataset.yaml` for Ultralytics.
2. **Fine-tune** — start from `yolo11n.pt` and train on the training split.
3. **Evaluate** — measure precision, recall, and mAP on validation or test data.
4. **Predict** — save annotated images and print the detected screw count for
   each image.
5. **Export (optional)** — export the trained weights to a deployment format.

## Setup

Run commands from this project directory in PowerShell:

```powershell
python -m pip install -r requirements.txt
```

The supplied configuration uses CPU training, so it works without a CUDA GPU.
To train on a supported NVIDIA GPU, install a CUDA-compatible PyTorch build and
set `train.device` in `config.yaml` to `"0"`.

## Run the project

```powershell
# Verify dataset layout and create dataset.yaml
python main.py --mode prepare

# Fine-tune YOLO11 (also prepares the dataset)
python main.py --mode train

# Evaluate the trained model
python main.py --mode val --split val
python main.py --mode val --split test

# Predict on the configured test images, or pass another image/folder/video
python main.py --mode predict
python main.py --mode predict --source "path\to\image-or-folder"

# Optional: export the fine-tuned model
python main.py --mode export --format onnx
```

Training settings, dataset locations, prediction options, and export settings
are in `config.yaml`. The best checkpoint is saved under
`runs/detect/runs/detect/yolov11_custom_train/weights/best.pt`; pass `--weights` to
evaluation, prediction, or export to select a different checkpoint.
