import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Date, Numeric, Boolean, PickleType
from base import Base
from session import session, engine

class Model(Base):
    __tablename__ = 'models'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    type = Column(String, nullable=False)
    model = Column(PickleType, nullable=False)
    symbol = Column(String, nullable=False)
    created = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    def __repr__(self):
        '''
        '''
        return '<%s: %s>' % (self.type, self.name)
