import json
from typing import Union
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from ultralytics import YOLO
import numpy as np
from schemas.responses import ValidationResponse
import tensorflow as tf
from pathlib import Path
from PIL import Image
import io
from fastapi.responses import JSONResponse

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World, Built by Jenkins"}


VALID_CLASSES = {"cat", "dog"}
# Load a pre-trained YOLOv8 model
model = YOLO("models/yolov8s.onnx") 
conf_threshold = 0.5  

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
        if conf < conf_threshold:
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

@app.post("/images/{animalType}/extractcharacteristics")
def extract_characteristics(animalType: str, image: UploadFile = File(...)):

    top_k = 3


    if animalType.lower() != "cat" and animalType.lower() != "dog":
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
    preprocessed_image = preprocess_image(content, BACKBONE)

    if animalType.lower() == "cat":
        probs = catBreedClassifier.predict(preprocessed_image, verbose=0)[0]
        class_names = cat_breed_class_names
    else:  # dog
        probs = dogBreedClassifier.predict(preprocessed_image, verbose=0)[0]
        class_names = dog_breed_class_names

    sorted_idx = np.argsort(probs)[::-1]
    best_idx = int(sorted_idx[0])
    top_k_idx = sorted_idx[1: 1 + top_k]

    predictions = [
        {
            "rank":       int(rank + 2),
            "breed":      class_names[idx],
            "confidence": round(float(probs[idx]), 4),
            "percent":    f"{probs[idx]:.1%}",
        }
        for rank, idx in enumerate(top_k_idx)
    ]

    return JSONResponse({
        "top_prediction": class_names[best_idx],
        "confidence":     round(float(probs[best_idx]), 4),
        "top_k":          predictions,
    })