from pydantic import BaseModel


class ValidationResponse(BaseModel):
    detectedClass: str
    confidence: float