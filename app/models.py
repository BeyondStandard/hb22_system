from database import Base
from sqlalchemy import Column, Integer, PickleType, String


class Audio(Base):
    __tablename__ = 'audio'

    id = Column(Integer, primary_key=True, index=True)
    car_type = Column(String)
    probability = Column(PickleType)
    file_name = Column(String)
    audio_encoded = Column(String)
    
