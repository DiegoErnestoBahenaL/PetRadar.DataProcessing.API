import json
from typing import Union
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from ultralytics import YOLO
import numpy as np
from configs import Configs
from schemas.responses import (
    ConfigsResponse,
    ValidationResponse,
    CharacteristicsResponse,
    TopPrediction,
    ColorInfo
)
from color_pattern_analyzer import (
    extract_colors,
    extract_glcm_features,
    extract_lbp_features,
    classify_pattern
)
import tensorflow as tf
from pathlib import Path
from PIL import Image
import io


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World, Built by Jenkins"}


VALID_CLASSES = {"cat", "dog"}
# Load a pre-trained YOLOv8 model
model = YOLO("models/yolov8s-seg.onnx") 


CAT_BREED_CLASSIFIER_MODEL_PATH = Path("models/cat-breed-classifier/efficientnet-20clases-224px-f1_0.71.keras")
CAT_BREED_CLASSIFIER_CLASS_NAMES_PATH = Path("models/cat-breed-classifier/class_names.json")

DOG_BREED_CLASSIFIER_MODEL_PATH = Path("models/dog-breed-classifier/efficientnet-120razas-224px-f1_0.80.keras")
DOG_BREED_CLASSIFIER_CLASS_NAMES_PATH = Path("models/dog-breed-classifier/class_names.json")

with open(CAT_BREED_CLASSIFIER_CLASS_NAMES_PATH) as f:
    cat_breed_class_names = json.load(f)

with open(DOG_BREED_CLASSIFIER_CLASS_NAMES_PATH) as f:
    dog_breed_class_names = json.load(f)

IMG_SIZE = (224, 224)
BACKBONE = "efficientnet"

global config
#load initial configs from JSON file
with open("configs.json") as f:
    config_data = json.load(f)

config = Configs()
config.yolo_conf_threshold = config_data.get("yolo_conf_threshold")
config.top_k_breed_predictions = config_data.get("top_k_breed_predictions")

def get_preprocessing_fn(backbone: str):
    if backbone == "efficientnet":
        return tf.keras.applications.efficientnet.preprocess_input
    return tf.keras.applications.mobilenet_v2.preprocess_input


def preprocess_image(image_bytes: bytes, backbone: str) -> np.ndarray:
    preprocess = get_preprocessing_fn(backbone)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess(arr)
    return arr

catBreedClassifier = tf.keras.models.load_model(
    CAT_BREED_CLASSIFIER_MODEL_PATH, compile=False
)

dogBreedClassifier = tf.keras.models.load_model(
    DOG_BREED_CLASSIFIER_MODEL_PATH, compile=False
)

@app.get("/configs")
def get_configs() -> ConfigsResponse:
    return ConfigsResponse(
        yoloConfThreshold=config.yolo_conf_threshold,
        topKBreedPredictions=config.top_k_breed_predictions
    )

@app.put("/configs")
def update_configs(yolo_conf_threshold: Union[float, None] = None, top_k_breed_predictions: Union[int, None] = None):
    global config
       
    if yolo_conf_threshold is not None:
        if not (0.0 <= yolo_conf_threshold <= 1.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="yolo_conf_threshold must be between 0.0 and 1.0"
            )
        config.yolo_conf_threshold = yolo_conf_threshold
        
    if top_k_breed_predictions is not None:
        if top_k_breed_predictions < 1 or top_k_breed_predictions > 5:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="top_k_breed_predictions must be a positive integer between 1 and 5"
            )
        config.top_k_breed_predictions = top_k_breed_predictions

    # Write updated configs back to JSON file
    with open("configs.json", "w") as f:
        json.dump({
            "yolo_conf_threshold": config.yolo_conf_threshold,
            "top_k_breed_predictions": config.top_k_breed_predictions
        }, f, indent=4) 

    return {"message": "Configs updated successfully"}

@app.post("/images/validatecatordog")
def validate_cat_or_dog(image: UploadFile = File(...)) -> ValidationResponse:

    content = image.file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded (no bytes received)."
        )
    
    numpy_image = np.frombuffer(content, np.uint8)
    if numpy_image.size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty image buffer received."
        )

    img = cv2.imdecode(numpy_image, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid decodable image."
        )

    results = model.predict(source=img, verbose=False)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="No object detected in the image."
        )
    
    # detectar IDs y cajas
    names = result.names
    boxes = result.boxes

    # se busca el candidato con la área máxima que sea válida
    best_candidate = None
    best_area = 0


    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        cls_name = names[cls_id]

        # filtrado por confianza
        if conf < config.yolo_conf_threshold:
            continue

        # filtrado por clase válida
        if cls_name not in VALID_CLASSES:
            continue

        # se calcula el area
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        area = max(0, x2 - x1) * max(0, y2 - y1)

        # actualizar el mejor candidato
        if area > best_area:
            best_area = area
            best_candidate = {
                "class_name": cls_name,
                "conf": conf,
                "box": (int(x1), int(y1), int(x2), int(y2))
            }

    # descartar si no hay candidatos válidos
    if best_candidate is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="No object detected with sufficient confidence or valid class (cat/dog)."
        )

    return ValidationResponse(
        detectedClass=best_candidate["class_name"],
        confidence=best_candidate["conf"]
    )

@app.post("/images/{animalType}/extractcharacteristics", response_model=CharacteristicsResponse)
def extract_characteristics(animalType: str, image: UploadFile = File(...)) -> CharacteristicsResponse:

    species = animalType.lower()

    if species != "cat" and species != "dog":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Currently only 'cat' and 'dog' are supported for characteristic extraction."
        )

    content = image.file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded (no bytes received)."
        )

    # Decode image for YOLO segmentation and color/pattern analysis
    numpy_image = np.frombuffer(content, np.uint8)
    if numpy_image.size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty image buffer received."
        )

    img = cv2.imdecode(numpy_image, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid decodable image."
        )

    # Run YOLOv8s-seg to obtain segmentation mask
    results = model.predict(source=img, verbose=False)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No object detected in the image."
        )

    if result.masks is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No segmentation mask produced."
        )

    # Find the largest-area valid detection (cat/dog above confidence threshold)
    names = result.names
    boxes = result.boxes
    best_candidate = None
    best_area = 0
    best_box_idx = -1

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        cls_name = names[cls_id]

        # filtrado por confianza
        if conf < config.yolo_conf_threshold:
            continue

        # filtrado por clase válida
        if cls_name not in VALID_CLASSES:
            continue

        # se calcula el area
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        area = max(0, x2 - x1) * max(0, y2 - y1)

        # actualizar el mejor candidato
        if area > best_area:
            best_box_idx = i
            best_area = area
            best_candidate = {
                "class_name": cls_name,
                "conf": conf,
                "box": (int(x1), int(y1), int(x2), int(y2))
            }

        # descartar si no hay candidatos válidos
    if best_candidate is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="No object detected with sufficient confidence or valid class (cat/dog)."
        )

    # Extract binary mask at original image resolution
    mask_data = result.masks.data[best_box_idx].cpu().numpy()
    binary_mask = cv2.resize(
        mask_data, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    binary_mask = (binary_mask > 0.5).astype(np.uint8)

    # Breed classification (uses original image bytes)
    preprocessed_image = preprocess_image(content, BACKBONE)

    if species == "cat":
        probs = catBreedClassifier.predict(preprocessed_image, verbose=0)[0]
        class_names = cat_breed_class_names
    else:
        probs = dogBreedClassifier.predict(preprocessed_image, verbose=0)[0]
        class_names = dog_breed_class_names

    sorted_idx = np.argsort(probs)[::-1]
    best_breed_idx = int(sorted_idx[0])
    top_k_idx = sorted_idx[1: 1 + config.top_k_breed_predictions]

    predictions = [
        TopPrediction(
            rank=int(rank + 2),
            breed=class_names[idx],
            confidence=round(float(probs[idx]), 4),
        )
        for rank, idx in enumerate(top_k_idx)
    ]

    # Coat color extraction and pattern classification on masked image
    colors_result = extract_colors(img, binary_mask)
    glcm_features = extract_glcm_features(img, binary_mask)
    lbp_features = extract_lbp_features(img, binary_mask)
    pattern_result = classify_pattern(
        img, binary_mask, species=species,
        colors=colors_result, glcm=glcm_features, lbp=lbp_features,
    )

    colors_out = [
        ColorInfo(
            color=c["color"],
            proportion=c["proportion"]
        )
        for c in colors_result
    ]

    return CharacteristicsResponse(
        topPredictedBreed=class_names[best_breed_idx],
        colors=colors_out,
        pattern=pattern_result,
        confidence=round(float(probs[best_breed_idx]), 4),
        topPredictions=predictions,
    )