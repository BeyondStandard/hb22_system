from sqlalchemy.orm import Session

from schemas import AudioCreate
from models import Audio as AudioModel
from mock import mock_audio


def get_audios(db: Session, skip: int = 0, limit: int = 100):
    return db.query(AudioModel).offset(skip).limit(limit).all()

def create_audio(db: Session, audio: AudioCreate):
    new_audio = AudioModel(audio_encoded = audio.audio_encoded, file_name = audio.file_name)
    db.add(new_audio)
    db.commit()
    db.refresh(new_audio)
    return new_audio

def get_latest_audio(db: Session):
    return db.query(AudioModel).order_by(AudioModel.id.desc()).first()