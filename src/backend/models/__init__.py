from .bi_mambaformer import BiMambaFormer, BidirectionalMambaBlock, MultiHeadFlowAttention

# Database models re-export for SQLAlchemy compatibility
try:
    from sqlalchemy import Column, Integer, String, Boolean, Float
    from ..database import Base

    class User(Base):
        __tablename__ = "users"
        id = Column(Integer, primary_key=True, index=True)
        email = Column(String, unique=True, index=True)
        hashed_password = Column(String)
        is_active = Column(Boolean, default=True)
        full_name = Column(String)
        oauth_provider = Column(String, nullable=True)
        oauth_id = Column(String, nullable=True)

    class Alert(Base):
        """Alert lifecycle entity supporting real-time SOC analyst workflow."""
        __tablename__ = "alerts"
        id = Column(Integer, primary_key=True, index=True)
        timestamp = Column(String, index=True)
        src_ip = Column(String, index=True)
        dst_ip = Column(String, index=True)
        src_port = Column(Integer, nullable=True)
        dst_port = Column(Integer, nullable=True)
        proto = Column(Integer, default=6)
        attack_type = Column(String, index=True)
        threat_score = Column(Float, index=True)
        severity = Column(String, default="HIGH")
        status = Column(String, default="new", index=True)
        analyst_notes = Column(String, nullable=True)
        culprit_feature = Column(String, nullable=True)
        mitre_technique = Column(String, nullable=True)
except Exception:
    pass
