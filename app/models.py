import json
import sqlalchemy
from database import Base
from sqlalchemy import Column, Integer, PickleType, String, TypeDecorator
 
class StringType(TypeDecorator):
    impl = sqlalchemy.TEXT(200000)

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(value)
        return value
    
    def process_result_value(self, value, dialect):
        if value is not None:
            value = json.loads(value)
        return value
        

class Audio(Base):
    __tablename__ = 'audio'

    id = Column(Integer, primary_key=True, index=True)
    car_type = Column(String)
    probability = Column(StringType())
    file_name = Column(String)
    audio_encoded = Column(String)
  
