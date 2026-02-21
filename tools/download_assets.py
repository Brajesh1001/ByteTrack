import os
import shutil
import argparse
import gdown
import kagglehub
from loguru import logger

# Mapping of Model Names to their Google Drive File IDs
MODEL_GDIVE_IDS = {
    "bytetrack_x_mot17": "1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5",
    "bytetrack_l_mot17": "1XwfUuCBF4IgWBWK2H7oOhQgEj9Mrb3rz",
    "bytetrack_m_mot17": "11Zb0NN_Uu7JwUd9e6Nk8o2_EUfxWqsun",
    "bytetrack_s_mot17": "1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj",
    "bytetrack_tiny_mot17": "1LFAl14sql2Q5Y9aNFsX_OqsnIzUD_1ju",
    "bytetrack_nano_mot17": "1AoN2AxzVwOLM0gJ15bcwqZUpFjlDV1dX",
    "bytetrack_x_mot20": "1HX2_JpMOjOIj1Z9rJjoet9XNy_cCAs5U"
}

def download_dataset():
    logger.info("Starting MOT15 dataset download from Kaggle...")
    path = kagglehub.dataset_download('mdraselsarker/mot15-challenge-dataset')
    
    target_dir = os.path.join("datasets", "MOT15")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        
    # KaggleHub usually downloads to a cache, we want to move/link it to our datasets folder
    # Source structure: path/MOT15/train, path/MOT15/test
    source_mot15 = os.path.join(path, "MOT15")
    
    if os.path.exists(source_mot15):
        logger.info(f"Moving dataset from cache to {target_dir}...")
        for item in os.listdir(source_mot15):
            s = os.path.join(source_mot15, item)
            d = os.path.join(target_dir, item)
            if not os.path.exists(d):
                shutil.move(s, d)
        logger.info("Dataset download and setup complete.")
    else:
        logger.warning(f"Could not find MOT15 folder in downloaded path: {path}")

def download_model(model_name):
    if model_name not in MODEL_GDIVE_IDS:
        logger.error(f"Unknown model: {model_name}. Available models: {list(MODEL_GDIVE_IDS.keys())}")
        return

    os.makedirs("pretrained", exist_ok=True)
    file_id = MODEL_GDIVE_IDS[model_name]
    output = os.path.join("pretrained", f"{model_name}.pth.tar")
    
    if os.path.exists(output):
        logger.info(f"Model {model_name} already exists at {output}. Skipping.")
        return

    logger.info(f"Downloading model: {model_name}...")
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output, quiet=False)
    logger.info(f"Model saved to {output}")

def main():
    parser = argparse.ArgumentParser(description="Download ByteTrack Datasets and Models")
    parser.add_argument("--dataset", action="store_true", help="Download MOT15 dataset")
    parser.add_argument("--model", type=str, choices=list(MODEL_GDIVE_IDS.keys()) + ["all"], 
                        help="Download a specific model or 'all'")
    
    args = parser.parse_args()
    
    if not args.dataset and not args.model:
        parser.print_help()
        return

    if args.dataset:
        download_dataset()
        
    if args.model:
        if args.model == "all":
            for m in MODEL_GDIVE_IDS:
                download_model(m)
        else:
            download_model(args.model)

if __name__ == "__main__":
    main()
