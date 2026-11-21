"""OpenCV Zoo modellarini models/ papkasiga yuklab oladi (~340 MB, bir marta)."""
import os
import urllib.request

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models/"
MODELS = {
    "face_detection_yunet_2023mar.onnx": "face_detection_yunet/",
    "face_recognition_sface_2021dec.onnx": "face_recognition_sface/",
    "object_detection_yolox_2022nov.onnx": "object_detection_yolox/",
    "person_reid_youtu_2021nov.onnx": "person_reid_youtureid/",
}
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(DIR, exist_ok=True)
for name, sub in MODELS.items():
    dst = os.path.join(DIR, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 100_000:
        print("bor:", name)
        continue
    print("yuklanmoqda:", name)
    urllib.request.urlretrieve(BASE + sub + name, dst + ".part")
    os.replace(dst + ".part", dst)
# CLIP (atributlar: ko'zoynak, kepka, soqol...) - ~155 MB
CLIP = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/"
CLIP_FILES = ["onnx/vision_model_quantized.onnx", "onnx/text_model_quantized.onnx", "tokenizer.json"]
os.makedirs(os.path.join(DIR, "clip"), exist_ok=True)
for f in CLIP_FILES:
    dst = os.path.join(DIR, "clip", os.path.basename(f))
    if os.path.exists(dst) and os.path.getsize(dst) > 100_000:
        print("bor:", os.path.basename(f))
        continue
    print("yuklanmoqda:", os.path.basename(f))
    urllib.request.urlretrieve(CLIP + f, dst + ".part")
    os.replace(dst + ".part", dst)
print("Tayyor.")
