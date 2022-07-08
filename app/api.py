from statistics import mode
from typing import List
from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from crud import create_audio, get_latest_audio
from crud import get_audios, create_audio
from schemas import Audio as AudioSchema
from schemas import Classifier
from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base
import sys
sys.path.insert(1, './../')

from main import Audio, Spectrography, Model

Base.metadata.create_all(bind=engine)

app = FastAPI()
handler = Mangum(app)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency
def get_db():
    db = SessionLocal()
    try: 
        yield db
    finally:
        db.close()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/latest", response_model=AudioSchema)
def get_latest( db: Session = Depends(get_db)):
    return get_latest_audio(db)


@app.get("/audios", response_model=List[AudioSchema])
def get_audio(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    audios = get_audios(db, skip=skip, limit=limit)
    return audios

@app.post("/ingest", response_model=AudioSchema)
async def ingest_audio_b64(schema: AudioSchema, db: Session = Depends(get_db)):
    
    file_path = Audio.base64_to_filepath(schema.audio_encoded)
    new_audio = Audio(file_path)
    new_audio.preprocess()
    s = Spectrography(new_audio)
    s.spectro_augment()
    model = Model()

    model.initialize_from_file("./../Models/cloud_model_1.pt", False)
    result = model.classify(s, unsqueeze = True)
    print(result)

    classifier= {
        "probability": "25%",
        "car_type": "jeep"
    }

    return create_audio(db, schema, classifier)
