import sys
import os
import argparse

from src.backend.database import SessionLocal, engine
from src.backend import models, auth

models.Base.metadata.create_all(bind=engine)
db = SessionLocal()

parser = argparse.ArgumentParser(description="Provision administrator user (Copilot Issue 8)")
parser.add_argument("--email", default=os.getenv("ADMIN_EMAIL", "admin@admin.com"), help="Admin email")
parser.add_argument("--password", default=os.getenv("ADMIN_PASSWORD", "admin"), help="Admin password")
parser.add_argument("--name", default="System Administrator", help="Admin display name")
args = parser.parse_args()

admin_email = args.email
admin_password = args.password
admin_name = args.name

try:
    user = db.query(models.User).filter(models.User.email == admin_email).first()
    if not user:
        hashed_password = auth.get_password_hash(admin_password)
        admin_user = models.User(
            email=admin_email, 
            hashed_password=hashed_password, 
            full_name=admin_name, 
            is_active=True
        )
        db.add(admin_user)
        db.commit()
        print(f"[+] Admin user created successfully! Email: {admin_email}, Role: admin")
    else:
        print(f"[*] Admin user already exists with Email: {admin_email}")
        
    users = db.query(models.User).all()
    print("Registered users:", [(u.email, u.full_name) for u in users])
except Exception as e:
    print("Error:", e)
finally:
    db.close()
