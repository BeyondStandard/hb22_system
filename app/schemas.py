from pydantic import BaseModel

class Classifier(BaseModel):
    car_type: str
    probability: str

class AudioBase(BaseModel):
    audio_encoded: str
    file_name: str

class AudioCreate(AudioBase):
    pass

class Audio(AudioBase):
    car_type: str
    probability: str

    class Config: orm_mode = True