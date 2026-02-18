"""
ByteTrack Pretrained Model Downloader

This script helps download pretrained ByteTrack models from Google Drive.

Models available:
- bytetrack_nano_mot17: Lightweight model (0.90M params, 69.0 MOTA)
- bytetrack_tiny_mot17: Small model (5.03M params, 77.1 MOTA)
- bytetrack_s_mot17: Small model (79.2 MOTA)
- bytetrack_m_mot17: Medium model (87.0 MOTA)
- bytetrack_l_mot17: Large model (88.7 MOTA)
- bytetrack_x_mot17: Extra large model (90.0 MOTA)
- bytetrack_x_mot20: Extra large for MOT20 (93.4 MOTA)
"""

import os
import sys

# Model URLs from Google Drive
MODELS = {
    "bytetrack_nano_mot17": {
        "url": "https://drive.google.com/file/d/1AoN2AxzVwOLM0gJ15bcwqZUpFjlDV1dX/view?usp=sharing",
        "file_id": "1AoN2AxzVwOLM0gJ15bcwqZUpFjlDV1dX",
        "filename": "bytetrack_nano_mot17.pth.tar"
    },
    "bytetrack_tiny_mot17": {
        "url": "https://drive.google.com/file/d/1LFAl14sql2Q5Y9aNFsX_OqsnIzUD_1ju/view?usp=sharing",
        "file_id": "1LFAl14sql2Q5Y9aNFsX_OqsnIzUD_1ju",
        "filename": "bytetrack_tiny_mot17.pth.tar"
    },
    "bytetrack_s_mot17": {
        "url": "https://drive.google.com/file/d/1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj/view?usp=sharing",
        "file_id": "1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj",
        "filename": "bytetrack_s_mot17.pth.tar"
    },
    "bytetrack_m_mot17": {
        "url": "https://drive.google.com/file/d/11Zb0NN_Uu7JwUd9e6Nk8o2_EUfxWqsun/view?usp=sharing",
        "file_id": "11Zb0NN_Uu7JwUd9e6Nk8o2_EUfxWqsun",
        "filename": "bytetrack_m_mot17.pth.tar"
    },
    "bytetrack_l_mot17": {
        "url": "https://drive.google.com/file/d/1XwfUuCBF4IgWBWK2H7oOhQgEj9Mrb3rz/view?usp=sharing",
        "file_id": "1XwfUuCBF4IgWBWK2H7oOhQgEj9Mrb3rz",
        "filename": "bytetrack_l_mot17.pth.tar"
    },
    "bytetrack_x_mot17": {
        "url": "https://drive.google.com/file/d/1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5/view?usp=sharing",
        "file_id": "1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5",
        "filename": "bytetrack_x_mot17.pth.tar"
    },
    "bytetrack_x_mot20": {
        "url": "https://drive.google.com/file/d/1HX2_JpMOjOIj1Z9rJjoet9XNy_cCAs5U/view?usp=sharing",
        "file_id": "1HX2_JpMOjOIj1Z9rJjoet9XNy_cCAs5U",
        "filename": "bytetrack_x_mot20.pth.tar"
    }
}

def download_model(model_name, output_dir="pretrained"):
    """Download a model from Google Drive using gdown"""
    try:
        import gdown
    except ImportError:
        print("gdown is not installed. Installing...")
        os.system(f"{sys.executable} -m pip install gdown")
        import gdown
    
    if model_name not in MODELS:
        print(f"Error: Model '{model_name}' not found!")
        print(f"Available models: {', '.join(MODELS.keys())}")
        return False
    
    model_info = MODELS[model_name]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, model_info["filename"])
    
    if os.path.exists(output_path):
        print(f"Model already exists: {output_path}")
        return True
    
    print(f"Downloading {model_name}...")
    print(f"URL: {model_info['url']}")
    print(f"Output: {output_path}")
    
    try:
        gdown.download(
            f"https://drive.google.com/uc?id={model_info['file_id']}",
            output_path,
            quiet=False
        )
        print(f"Successfully downloaded: {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def list_models():
    """List all available models"""
    print("\nAvailable ByteTrack Models:")
    print("=" * 70)
    for name in MODELS.keys():
        print(f"  - {name}")
    print("=" * 70)
    print("\nUsage: python download_model.py <model_name>")
    print("Example: python download_model.py bytetrack_nano_mot17")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        list_models()
    else:
        model_name = sys.argv[1]
        download_model(model_name)
