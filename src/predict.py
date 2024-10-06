import os
import zipfile
from datasets import load_dataset, Dataset, Features, ClassLabel, Image, concatenate_datasets, Array3D
import torch
from transformers import ViTFeatureExtractor, AutoModelForImageClassification, TrainingArguments, Trainer
from PIL import Image as PILImage
import logging
from datasets import config, DownloadConfig
import pandas as pd
import numpy as np
import ssl
import certifi
import urllib.request

# Global variables
MODEL_DIR = "./trained_model"
BATCH_SIZE = 7  # Adjust as needed based on your system's memory
NUM_WORKERS = 0  # Set to 0 to disable multiprocessing
NUM_EPOCHS = 6
config.HF_DATASETS_CACHE = "./cache"

# Set up logging
logging.basicConfig(level=logging.INFO, filename='preprocess.log', filemode='w')

# Download configuration
download_config = DownloadConfig(
    max_retries=10,
    num_proc=1
)

# Clear cache function
def clear_cache():
    import shutil
    cache_dir = config.HF_DATASETS_CACHE
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cache directory '{cache_dir}' has been deleted.")

# Unzip Bottle Cap Dataset
def unzip_bottlecap_dataset():
    zip_path = "./bottle-cap-classification.zip"
    extract_path = "./cache/tahuuanh__bottle_cap_classification"
    if not os.path.exists(extract_path):
        print("Unzipping the Bottle Cap dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

# Load Bottle Cap Dataset
def load_bottlecap_dataset():
    bottlecap_train_path = "./cache/tahuuanh__bottle_cap_classification/nap chai/Train"
    bottlecap_val_path = "./cache/tahuuanh__bottle_cap_classification/nap chai/Val"

    # Create lists to store file paths and labels
    train_filepaths, train_labels = [], []
    val_filepaths, val_labels = [], []

    # Load train data
    class_names = sorted(os.listdir(bottlecap_train_path))  # Get class names from directory
    # Map bottle cap labels to existing labels
    label_mapping = {
        'nhua': 'plastic',
        'kim_loai': 'metal',
        # Add other mappings if necessary
    }
    for class_name in class_names:
        mapped_label = label_mapping.get(class_name, 'trash')  # Default to 'trash' if not mapped
        class_path = os.path.join(bottlecap_train_path, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                train_filepaths.append(os.path.join(class_path, img_name))
                train_labels.append(mapped_label)

    # Load validation data
    for class_name in class_names:
        mapped_label = label_mapping.get(class_name, 'trash')  # Default to 'trash' if not mapped
        class_path = os.path.join(bottlecap_val_path, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                val_filepaths.append(os.path.join(class_path, img_name))
                val_labels.append(mapped_label)

    # Create pandas DataFrames
    train_df = pd.DataFrame({"image": train_filepaths, "label": train_labels})
    val_df = pd.DataFrame({"image": val_filepaths, "label": val_labels})

    # Convert to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # Define the features with consistent class names
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    features = Features({
        "image": Image(decode=True),
        "label": ClassLabel(names=class_names)
    })
    train_dataset = train_dataset.cast(features)
    val_dataset = val_dataset.cast(features)

    return train_dataset, val_dataset

# Preprocess function for the dataset
def preprocess(example, idx=None):
    try:
        print(f"Processing image at index {idx}", flush=True)
        image = example["image"]

        if image is None:
            print(f"Image at index {idx} is None.", flush=True)
            return {}

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize image to reduce memory usage
        max_size = (500, 500)
        image = image.resize(max_size)

        # Use feature_extractor to get numpy arrays
        inputs = feature_extractor(images=image, return_tensors="np")

        # Get the pixel values numpy array
        pixel_values = inputs["pixel_values"][0]  # shape (3, 224, 224)
        labels = example["label"]

        result = {"pixel_values": pixel_values, "labels": labels}

        # Clear variables to free memory
        del image
        del inputs

        return result
    except Exception as e:
        print(f"Error processing image at index {idx}: {e}", flush=True)
        return {}

# Preprocess and filter datasets
def preprocess_and_filter(dataset, label_names):
    # Process the dataset sequentially without multiprocessing
    processed_examples = []
    for idx, example in enumerate(dataset):
        result = preprocess(example, idx)
        if "pixel_values" in result and "labels" in result:
            processed_examples.append(result)
        else:
            print(f"Skipping example at index {idx}", flush=True)
    # Define the features explicitly
    features = Features({
        "pixel_values": Array3D(dtype="float32", shape=(3, 224, 224)),
        "labels": ClassLabel(names=label_names)
    })
    return Dataset.from_dict({
        "pixel_values": [ex["pixel_values"] for ex in processed_examples],
        "labels": [ex["labels"] for ex in processed_examples]
    }, features=features)

# Function to load and combine datasets
def load_and_combine_datasets():
    # Load Trash Classification Dataset
    dataset_cache_path = "./cache/ethanwan__trash_classification"
    if os.path.exists(dataset_cache_path):
        print("Loading the cached Trash Classification dataset...")
        trash_dataset = load_dataset("ethanwan/trash_classification", cache_dir="./cache")
    else:
        print("Downloading Trash Classification dataset...")
        trash_dataset = load_dataset("ethanwan/trash_classification", download_config=download_config, cache_dir="./cache")
    train_test_split1 = trash_dataset["train"].train_test_split(test_size=0.2)
    trash_train, trash_eval = train_test_split1["train"], train_test_split1["test"]

    # Load Bottle Cap Dataset
    bottlecap_train_dataset, bottlecap_val_dataset = load_bottlecap_dataset()

    # Define consistent class names
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    label_feature = ClassLabel(names=class_names)

    # Update ClassLabel features for all datasets
    trash_train = trash_train.cast_column("label", label_feature)
    trash_eval = trash_eval.cast_column("label", label_feature)
    bottlecap_train_dataset = bottlecap_train_dataset.cast_column("label", label_feature)
    bottlecap_val_dataset = bottlecap_val_dataset.cast_column("label", label_feature)

    # Combine datasets
    combined_train = concatenate_datasets([trash_train, bottlecap_train_dataset])
    combined_eval = concatenate_datasets([trash_eval, bottlecap_val_dataset])

    return combined_train, combined_eval

# Check if a trained model exists
def check_saved_model():
    global feature_extractor
    if os.path.exists(MODEL_DIR):
        print("Loading saved model...")
        model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
        feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_DIR)
    else:
        print("No saved model found. Training a new model...")
        feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        # Load and combine datasets
        train_dataset, eval_dataset = load_and_combine_datasets()
        train_model(train_dataset, eval_dataset)
        # Load the trained model
        model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
        feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_DIR)
    return model, feature_extractor

# Train the model
def train_model(train_dataset, eval_dataset):
    # Retrieve label names from the original dataset
    label_names = train_dataset.features["label"].names
    # Preprocess the datasets
    train_dataset = preprocess_and_filter(train_dataset, label_names)
    eval_dataset = preprocess_and_filter(eval_dataset, label_names)

    # Initialize the model
    num_labels = len(label_names)
    model = AutoModelForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )

    # Set label mappings
    model.config.label2id = {label: i for i, label in enumerate(label_names)}
    model.config.id2label = {i: label for i, label in enumerate(label_names)}

    # Set the device to MPS if available (for Apple Silicon), else CPU
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    model.to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        save_steps=1000000,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=NUM_WORKERS,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    trainer.train()
    model.save_pretrained(MODEL_DIR)
    feature_extractor.save_pretrained(MODEL_DIR)

# Collate function to handle batches properly
def collate_fn(batch):
    pixel_values = torch.tensor([item["pixel_values"] for item in batch], dtype=torch.float32)
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "labels": labels
    }

# Function to split the image into patches
def split_image(image, patch_size, overlap=0.1):
    width, height = image.size
    step = int(patch_size * (1 - overlap))
    patches = []
    for top in range(0, height, step):
        for left in range(0, width, step):
            box = (left, top, min(left + patch_size, width), min(top + patch_size, height))
            patch = image.crop(box)
            patches.append((patch, box))
    return patches

# Function to process each patch for prediction
def predict_patch(patch, model, feature_extractor, threshold=0.4):
    inputs = feature_extractor(images=patch, return_tensors="pt")

    # Set the device to MPS if available
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    max_prob, predicted_class_idx = torch.max(probs, dim=-1)

    if max_prob.item() < threshold:
        predicted_label = "no labels"
    else:
        predicted_label = model.config.id2label[predicted_class_idx.item()]

    return predicted_label, max_prob.item()

def detect_trash(image_path, model, feature_extractor, threshold=0.4):
    image = PILImage.open(image_path).convert("RGB")
    # Resize image to model's expected input size
    image = image.resize((224, 224))
    inputs = feature_extractor(images=image, return_tensors="pt")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    max_prob, predicted_class_idx = torch.max(probs, dim=-1)
    if max_prob.item() < threshold:
        predicted_label = "no labels"
    else:
        predicted_label = model.config.id2label[predicted_class_idx.item()]
    if predicted_label == 'glass':
        predicted_label='plastic' #common mistake by the learner
    print(f"Predicted Label: {predicted_label} with confidence: {max_prob.item():.2f}")
    return predicted_label

def predict_main(img):
    # Ensure the Bottle Cap dataset is unzipped
    unzip_bottlecap_dataset()

    # Check for a saved model or train a new one
    model, feature_extractor = check_saved_model()

    # Perform detection on a sample image
    final_label = detect_trash(
        img,
        model,
        feature_extractor
    )
    print(final_label)
    return final_label

# Example usage
if __name__ == "__main__":
    # Ensure the Bottle Cap dataset is unzipped
    unzip_bottlecap_dataset()


    # Check for a saved model or train a new one
    model, feature_extractor = check_saved_model()

    # Perform detection on a sample image
    final_label = detect_trash(
        "/your/path.png",
        model,
        feature_extractor
    )