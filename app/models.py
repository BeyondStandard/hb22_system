from database import Base
from sqlalchemy import Column, Integer, String


class Audio(Base):
    __tablename__ = 'audio'

    id = Column(Integer, primary_key=True, index=True)
    audio_encoded = Column(String)
    image = Column(String)
    
