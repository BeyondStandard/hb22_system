import asyncio
from statistics import mode
from typing import Any, List, Dict, Optional
from fastapi import Depends, FastAPI, File, UploadFile, WebSocket
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

ingest_state = False

app = FastAPI()
handler = Mangum(app)

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    global ingest_state
    ingest_state = False

async def set_state():
    await asyncio.sleep(3)
    global ingest_state
    ingest_state = True

async def get_state():
    await asyncio.sleep(3)
    global ingest_state
    return ingest_state

# Dependency
def get_db():
    db = SessionLocal()
    try: 
        yield db
    finally:
        db.close()


async def handler(websocket, path):
    reply = "lets go"
    await websocket.send(reply)

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
    
    model = Model()
    model.initialize_from_file("./../Models/cloud_with_electric.pt", True)
    result = model.server_process(schema.audio_encoded)

    classifier= {
        "probability": result,
        "car_type": "jeep"
    }

    returnvalue =  create_audio(db, schema, classifier)
    await set_state()
    return returnvalue


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global ingest_state
    print('Accepting client connection...')
    await websocket.accept()

    while True:
        try:
            global ingest_state
            # Send message to the client
            result = await get_state()
            if result is True:
                latest_audio = get_latest_audio(get_db())
                print(latest_audio)
                resp = {"state": ingest_state}
                asyncio.io.sleep(150)
                await websocket.send_json(resp)
                ingest_state = False
        except Exception as e:
            print('error:', e)
            break
    print('Bye..')

