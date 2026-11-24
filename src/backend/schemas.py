from pydantic import BaseModel
from typing import Optional

class UserBase(BaseModel):
    email: str
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class AlertBase(BaseModel):
    timestamp: str
    src_ip: str
    dst_ip: str
    src_port: Optional[int] = None
    dst_port: Optional[int] = None
    proto: int = 6
    attack_type: str
    threat_score: float
    severity: str = "HIGH"
    status: str = "new"
    analyst_notes: Optional[str] = None
    culprit_feature: Optional[str] = None
    mitre_technique: Optional[str] = None

class AlertCreate(AlertBase):
    pass

class AlertUpdate(BaseModel):
    status: Optional[str] = None
    analyst_notes: Optional[str] = None

class AlertResponse(AlertBase):
    id: int

    class Config:
        from_attributes = True
