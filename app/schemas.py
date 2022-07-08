from typing import Union
from pydantic import BaseModel

class Classifier(BaseModel):
    car_type: str
    probability: Union[str, dict]

class AudioBase(BaseModel):
    audio_encoded: str
    file_name: str

class AudioCreate(AudioBase):
    pass

class Audio(AudioBase):
    car_type: str
    probability: Union[str, dict]

    class Config: orm_mode = True