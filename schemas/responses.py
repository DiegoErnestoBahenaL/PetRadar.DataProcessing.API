from pydantic import BaseModel


class ValidationResponse(BaseModel):
    detectedClass: str
    confidence: float

class TopPrediction(BaseModel):
    rank: int
    breed: str
    confidence: float

class CharacteristicsResponse(BaseModel):
    topPredictedBreed: str
    confidence: float
    topPredictions: list[TopPrediction]
