from pydantic import BaseModel

class ValidationResponse(BaseModel):
    detectedClass: str
    confidence: float

class TopPrediction(BaseModel):
    rank: int
    breed: str
    confidence: float

class ColorInfo(BaseModel):
    color: str
    proportion: float


class CharacteristicsResponse(BaseModel):
    topPredictedBreed: str
    colors: list[ColorInfo]
    pattern: str
    confidence: float
    topPredictions: list[TopPrediction]