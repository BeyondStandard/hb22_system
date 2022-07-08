from pydantic import BaseModel


class AudioBase(BaseModel):
    audio_encoded: str
    image: str

class AudioCreate(AudioBase):
    pass

class Audio(AudioBase):

    class Config: orm_mode = True