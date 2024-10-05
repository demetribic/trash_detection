import os
from datasets import load_dataset, concatenate_datasets
import torch
from transformers import ViTFeatureExtractor, AutoModelForImageClassification, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from PIL import Image
import signal
from datasets import config, DownloadConfig

# Set a larger timeout duration in seconds
download_config = DownloadConfig(
    max_retries=5,  # Increases the number of retries
    num_proc=4    # Number of parallel downloads

)

# Load dataset with modified config


config.HF_DATASETS_CACHE = "./cache"

# Global variables
MODEL_DIR = "./trained_model"
BATCH_SIZE = 8
NUM_WORKERS = 2
NUM_EPOCHS = 3
def clear_cache():
    import shutil
    cache_dir = config.HF_DATASETS_CACHE
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cache directory '{cache_dir}' has been deleted.")
# Signal handler for image loading timeout
def handler(signum, frame):
    raise Exception("Image loading timeout")

# Preprocess function for the dataset
def preprocess(example, idx=None):
    try:
        print(f"Processing image at index {idx}")
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(5)  # Set the timeout to 5 seconds

        image = example["image"]

        if isinstance(image, str):
            # If the image is a file path, open it
            image = Image.open(image)

        # Disable the alarm after successful image loading
        signal.alarm(0)

        # Convert image to RGB to ensure consistent format
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Extract features using the feature extractor
        inputs = feature_extractor(images=image, return_tensors="pt")
        inputs["labels"] = example["label"]

        return inputs
    except OSError as e:
        print(f"Skipping corrupt image at index {idx}: {e}")
        return {}
    except Exception as e:
        print(f"Error at index {idx}: {e}")
        return {}

# Filter function to remove invalid images
def filter_invalid(example):
    return bool(example)

# Preprocess and filter datasets
def preprocess_and_filter(dataset):
    dataset = dataset.map(lambda x, idx: preprocess(x, idx), with_indices=True, batched=False)
    dataset = dataset.filter(filter_invalid)
    import shutil
    shutil.rmtree(dataset.cache_files)
    return dataset

# Collate function to handle batches properly
def collate_fn(batch):
    pixel_values = [torch.tensor(x["pixel_values"]) if isinstance(x["pixel_values"], list) else x["pixel_values"] for x in batch]
    return {
        "pixel_values": torch.stack([x.squeeze() for x in pixel_values]),
        "labels": torch.tensor([x["labels"] for x in batch])
    }

# Create DataLoader for training and evaluation
def create_dataloaders(train_dataset, eval_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    return train_dataloader, eval_dataloader



def load_datasets():
    # Load your original dataset
    dataset1 = load_dataset("ethanwan/trash_classification")
    train_test_split1 = dataset1["train"].train_test_split(test_size=0.2)

    # Load the COCO dataset and sample only 5% of it
    coco_dataset = load_dataset("shunk031/MSCOCO", year=2017, coco_task="instances", decode_rle=True)
    
    # Sample 5% of the COCO dataset
    coco_sampled = coco_dataset.train_test_split(test_size=0.95, seed=42)['train']

    # Filter out relevant categories from COCO (like bottles, plastic bags, etc.)
    def filter_coco(example):
        relevant_labels = [44, 46]  # COCO label IDs for 'bottle', 'cup', etc.
        return example["category_id"] in relevant_labels
    
    coco_sampled = coco_sampled.filter(filter_coco)
    
    # Concatenate the datasets
    train_dataset = concatenate_datasets([train_test_split1["train"], coco_sampled])
    eval_dataset = concatenate_datasets([train_test_split1["test"], coco_sampled])

    # Preprocess and filter the combined datasets
    train_dataset = preprocess_and_filter(train_dataset)
    eval_dataset = preprocess_and_filter(eval_dataset)
    
    return train_dataset, eval_dataset
    
# Train the model
def train_model(train_dataset, eval_dataset):
    model = AutoModelForImageClassification.from_pretrained(
        "google/vit-base-patch16-224", 
        ignore_mismatched_sizes=True, 
        num_labels=len(train_dataset.features["label"].names)  # Access features correctly
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        save_steps=1000000,  # Set save_steps to a high value to avoid saving
        save_total_limit=None,
        logging_dir="./logs",
        load_best_model_at_end=False,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        optimizers=(None, None)
    )

    # Train the model
    trainer.train()

    # Save the trained model
    model.save_pretrained(MODEL_DIR)
    feature_extractor.save_pretrained(MODEL_DIR)

    # Save the label names to the model's config
    label_names = train_dataset.features["label"].names  # Correct label access
    model.config.label2id = {label: i for i, label in enumerate(label_names)}
    model.config.id2label = {i: label for i, label in enumerate(label_names)}
    model.save_pretrained(MODEL_DIR)  # Re-save the model with updated config

# Check if a trained model exists
def check_saved_model():
    global feature_extractor  # Declare global feature_extractor
    if os.path.exists(MODEL_DIR):
        print("Loading saved model...")
        model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
        feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_DIR)
    else:
        print("No saved model found. Training a new model...")
        feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        train_dataset, eval_dataset = load_datasets()
        train_model(train_dataset, eval_dataset)
        model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
        feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_DIR)
    return model, feature_extractor

# Preprocess a single image for prediction
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs

# Function to split the image into patches
def split_image(image, patch_size):
    width, height = image.size
    patches = []
    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            box = (left, top, left + patch_size, top + patch_size)
            patch = image.crop(box)
            patches.append((patch, box))  # Return both patch and the coordinates
    return patches

# Function to process each patch for prediction
def predict_patch(patch, model, feature_extractor, threshold=0.5):
    inputs = feature_extractor(images=patch, return_tensors="pt")

    # Move inputs to the same device as the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Put the model in evaluation mode
    model.eval()

    # Perform the prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the logits and apply softmax to get probabilities
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
    max_prob, predicted_class_idx = torch.max(probs, dim=-1)

    # Check if the highest probability is above the threshold
    if max_prob.item() < threshold:
        predicted_label = "no labels"  # Output "no labels" if not confident
    else:
        predicted_label = model.config.id2label[predicted_class_idx.item()]
    
    return predicted_label, max_prob.item()

# Function to process the entire image in patches
def predict_image_patches(image_path, model, feature_extractor, patch_size=336, threshold=0.4):
    image = Image.open(image_path).convert("RGB")
    patches = split_image(image, patch_size)
    trash_patches = []

    for patch, box in patches:
        predicted_label, confidence = predict_patch(patch, model, feature_extractor, threshold)
        if predicted_label != "no labels":
            print(f"Detected {predicted_label} with confidence {confidence} in patch {box}")
            if predicted_label:
                trash_patches.append((box, confidence, predicted_label))
    
    return trash_patches

# Use the updated function with a larger patch size and lower threshold
if __name__ == "__main__":
    # Check if a saved model exists and load it
    model, feature_extractor = check_saved_model()

    # Test the function with an image file, using larger patches (e.g., 336x336)
    trash_patches = predict_image_patches("/Users/demetri/Documents/GitHub/trash_detection/IMG_1324.jpg", model, feature_extractor)

    if trash_patches:
        print("Instances of trash/plastic detected in the following patches:")
        for box, confidence, label in trash_patches:
            print(f"Patch: {box}, Confidence: {confidence}, Label: {label}")
    else:
        print("No instances of trash or plastic detected.")
    clear_cache()