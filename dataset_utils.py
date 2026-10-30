import yaml
from pathlib import Path


def inspect_dataset_folder(folder_path):
    """
    Inspects a dataset split folder (train, val/valid, test) for images and labels.
    Supports both subfolder structure (images/, labels/) and flat structure.
    """
    folder = Path(folder_path)
    if not folder.exists():
        return {"images": 0, "labels": 0, "images_dir": None, "labels_dir": None, "exists": False}

    if folder.name.lower() == "images" and folder.is_dir():
        images_dir = folder
        labels_dir = folder.parent / "labels"
    else:
        images_dir = folder / "images" if (folder / "images").is_dir() else folder
        labels_dir = folder / "labels" if (folder / "labels").is_dir() else folder

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    ] if images_dir.is_dir() else []
    label_files = list(labels_dir.glob("*.txt")) if labels_dir.is_dir() else []
    label_stems = {path.stem for path in label_files}
    missing_labels = [path.name for path in image_files if path.stem not in label_stems]

    return {
        "images": len(image_files),
        "labels": len(label_files),
        "images_dir": images_dir.resolve(),
        "labels_dir": labels_dir.resolve(),
        "exists": True,
        "missing_labels": missing_labels,
    }


def prepare_dataset_yaml(config_data, output_yaml_path="dataset.yaml"):
    """
    Prepares and validates dataset directories (train, valid, test) and generates
    the dataset YAML configuration required by YOLOv11.
    """
    dataset_cfg = config_data.get("dataset", {})
    dataset_dir = Path(dataset_cfg.get("dataset_dir", ".")).resolve()

    train_path_cfg = dataset_cfg.get("train_path", "train")
    val_path_cfg = dataset_cfg.get("val_path", "valid")
    test_path_cfg = dataset_cfg.get("test_path", "test")

    # Resolve full paths
    train_path_cfg = Path(train_path_cfg)
    train_full = dataset_dir / train_path_cfg if not train_path_cfg.is_absolute() else train_path_cfg

    # Try alternate valid naming (valid or val)
    val_path_cfg = Path(val_path_cfg)
    val_full = dataset_dir / val_path_cfg if not val_path_cfg.is_absolute() else val_path_cfg
    if not val_full.exists() and "valid" in str(val_full):
        alt_val = dataset_dir / str(val_path_cfg).replace("valid", "val")
        if alt_val.exists():
            val_full = alt_val
    elif not val_full.exists() and "val" in str(val_full):
        alt_val = dataset_dir / str(val_path_cfg).replace("val", "valid")
        if alt_val.exists():
            val_full = alt_val

    test_path_cfg = Path(test_path_cfg)
    test_full = dataset_dir / test_path_cfg if not test_path_cfg.is_absolute() else test_path_cfg

    # Inspect splits
    train_stats = inspect_dataset_folder(train_full)
    val_stats = inspect_dataset_folder(val_full)
    test_stats = inspect_dataset_folder(test_full)

    print("=" * 60)
    print("📊 DATASET INSPECTION SUMMARY")
    print("=" * 60)
    print(f"Dataset Root: {dataset_dir}")
    print(f"• Train Set : {train_stats['images']} images, {train_stats['labels']} label files at {train_stats['images_dir']}")
    print(f"• Valid Set : {val_stats['images']} images, {val_stats['labels']} label files at {val_stats['images_dir']}")
    if test_stats['exists']:
        print(f"• Test Set  : {test_stats['images']} images, {test_stats['labels']} label files at {test_stats['images_dir']}")
    else:
        print("• Test Set  : (Optional / Not found)")
    print("=" * 60)

    if train_stats['images'] == 0:
        raise ValueError(f"No training images found in path: {train_full}")
    if val_stats['images'] == 0:
        raise ValueError(f"No validation images found in path: {val_full}")

    # Check for existing data.yaml inside dataset directory for class names auto-discovery
    class_names = dataset_cfg.get("names", ["object"])
    num_classes = dataset_cfg.get("nc", len(class_names))

    existing_data_yaml = dataset_dir / "data.yaml"
    if existing_data_yaml.exists():
        with open(existing_data_yaml, "r", encoding="utf-8") as f:
            existing_data = yaml.safe_load(f)
        if not isinstance(existing_data, dict):
            raise ValueError(f"Dataset YAML must contain a mapping: {existing_data_yaml}")
        class_names = existing_data.get("names", class_names)
        num_classes = existing_data.get("nc", len(class_names))
        print(f"ℹ️ Auto-detected class configuration from {existing_data_yaml.name}: {num_classes} class(es) -> {class_names}")

    if not isinstance(class_names, (list, dict)) or not class_names:
        raise ValueError("Dataset class names must be a non-empty list or mapping.")
    if num_classes != len(class_names):
        raise ValueError(
            f"Dataset class count mismatch: nc={num_classes}, but names contains {len(class_names)} class(es)."
        )

    for split_name, stats in (("train", train_stats), ("valid", val_stats), ("test", test_stats)):
        if stats["images"] and stats["missing_labels"]:
            print(
                f"⚠️ {split_name.capitalize()} set has {len(stats['missing_labels'])} image(s) "
                f"without matching label files."
            )
        if stats["images"]:
            for label_path in stats["labels_dir"].glob("*.txt"):
                with open(label_path, "r", encoding="utf-8") as f:
                    for line_number, line in enumerate(f, start=1):
                        fields = line.split()
                        if len(fields) != 5:
                            raise ValueError(
                                f"Invalid YOLO label row in {label_path}:{line_number}; expected 5 values."
                            )
                        try:
                            class_id = int(fields[0])
                            coordinates = [float(value) for value in fields[1:]]
                        except ValueError as exc:
                            raise ValueError(
                                f"Non-numeric YOLO label value in {label_path}:{line_number}."
                            ) from exc
                        if not 0 <= class_id < num_classes:
                            raise ValueError(
                                f"Class id {class_id} outside [0, {num_classes - 1}] in "
                                f"{label_path}:{line_number}."
                            )
                        if any(not 0 <= value <= 1 for value in coordinates):
                            raise ValueError(
                                f"YOLO coordinates must be between 0 and 1 in {label_path}:{line_number}."
                            )

    # Build dataset dictionary with absolute path references for YOLOv11
    yaml_content = {
        "path": str(dataset_dir),
        "train": str(train_stats['images_dir']) if train_stats['images_dir'] else str(train_full),
        "val": str(val_stats['images_dir']) if val_stats['images_dir'] else str(val_full),
        "nc": num_classes,
        "names": class_names
    }

    if test_stats['exists'] and test_stats['images_dir']:
        yaml_content["test"] = str(test_stats['images_dir'])

    target_yaml_path = Path(output_yaml_path).resolve()
    target_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"✅ Generated dataset configuration file at: {target_yaml_path}\n")
    return str(target_yaml_path)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    prepare_dataset_yaml(cfg)
