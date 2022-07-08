from sqlalchemy.orm import Session

from schemas import AudioCreate
from models import Audio as AudioModel
from mock import mock_audio


def get_audios(db: Session, skip: int = 0, limit: int = 100):
    return db.query(AudioModel).offset(skip).limit(limit).all()

def create_audio(db: Session, audio: AudioCreate, classifier):
    new_audio = AudioModel(car_type= classifier["car_type"], probability = classifier["probability"], audio_encoded=audio.audio_encoded, file_name = audio.file_name)
    print(new_audio)
    db.add(new_audio)
    db.commit()
    db.refresh(new_audio)
    new_audio.probability = dict()
    return new_audio

def get_latest_audio(db: Session):
    return db.query(AudioModel).order_by(AudioModel.id.desc()).first()