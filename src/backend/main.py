import os
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from authlib.integrations.starlette_client import OAuth
from datetime import timedelta
from typing import List
import asyncio
import threading
import json
import queue

from . import models, schemas, auth, database
from .database import engine, get_db
from .sniffer import RealTimeFeatureExtractor, start_sniffing
from .detector import HybridDetector
from .rag_service import RAGExplanationService

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Hybrid NIDS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_origin_regex=r"http://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SessionMiddleware, secret_key=auth.SECRET_KEY)

# OAuth Setup
oauth = OAuth()
import os
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name='google',
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid email profile'}
    )

# Resolve model directory relative to this file (works from any CWD)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_BASE_DIR, "saved_models")

# Global instances
detector = HybridDetector(_MODELS_DIR)
extractor = RealTimeFeatureExtractor(models_dir=_MODELS_DIR)
rag_service = RAGExplanationService()

# WebSocket clients
connected_clients = set()
# Background processing queue
packet_queue = queue.Queue()

@app.post("/signup", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = auth.get_password_hash(user.password)
    db_user = models.User(email=user.email, hashed_password=hashed_password, full_name=user.full_name)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/login/google")
async def login_google(request: Request):
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=400, detail="Google OAuth not configured")
    redirect_uri = request.url_for('auth_google')
    return await oauth.google.authorize_redirect(request, str(redirect_uri))

@app.get("/auth/google")
async def auth_google(request: Request, db: Session = Depends(get_db)):
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception:
        raise HTTPException(status_code=401, detail="Google auth failed")
        
    user_info = token.get('userinfo')
    if not user_info:
        raise HTTPException(status_code=401, detail="No user info from Google")
        
    email = user_info['email']
    user = db.query(models.User).filter(models.User.email == email).first()
    
    if not user:
        # Create new user via OAuth
        user = models.User(
            email=email,
            full_name=user_info.get('name', ''),
            oauth_provider='google',
            oauth_id=user_info.get('sub'),
            is_active=True
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    # Redirect to frontend with token
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"http://localhost:5173/login?token={access_token}")

@app.post("/token", response_model=schemas.Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # 1. Allow default admin credentials (admin / admin or admin@nids.local / admin)
    if form_data.username.lower() in ("admin", "admin@nids.local", "admin@hybrid.local") and form_data.password in ("admin", "admin123"):
        user = db.query(models.User).filter(models.User.email.in_(["admin@hybrid.local", "admin@nids.local", "admin"])).first()
        if not user:
            user = models.User(
                email="admin@hybrid.local",
                full_name="System Administrator",
                hashed_password=auth.get_password_hash("admin"),
                is_active=True
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES * 48)
        access_token = auth.create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}

    # 2. Standard user lookup
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=schemas.User)
async def read_users_me(current_user: models.User = Depends(auth.get_current_user)):
    return current_user

from .traffic_generator import generator_instance
from fastapi import UploadFile, File
from pydantic import BaseModel

class CaptureModeRequest(BaseModel):
    mode: str # 'live', 'virtual_synthetic', 'physical_synthetic'

@app.post("/api/capture/mode")
async def set_capture_mode(request: CaptureModeRequest, current_user: models.User = Depends(auth.get_current_user)):
    mode = request.mode
    if mode == 'live':
        generator_instance.stop()
        return {"status": "success", "message": "Live network sniffing active. Synthetic generator stopped."}
    elif mode == 'virtual_synthetic':
        generator_instance.start(virtual=True, target_queue=packet_queue)
        return {"status": "success", "message": "Virtual synthetic injection started."}
    elif mode == 'physical_synthetic':
        generator_instance.start(virtual=False, target_queue=packet_queue)
        return {"status": "success", "message": "Physical synthetic injection started."}
    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

import pandas as pd
import io
import time

class PolicyRequest(BaseModel):
    policy: str  # 'p95', 'p99', 'youden', 'fpr_3pct'

@app.post("/api/policy")
async def update_threshold_policy(req: PolicyRequest, current_user: models.User = Depends(auth.get_current_user)):
    success = detector.set_policy(req.policy)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid policy. Available options: {list(detector.threshold_policies.keys())}"
        )
    return {
        "status": "success",
        "policy": req.policy,
        "threshold": detector.base_threshold,
        "message": f"Active policy changed to '{req.policy}' (Operating Threshold: {detector.base_threshold:.6f})"
    }

@app.get("/api/policy")
async def get_threshold_policies(current_user: models.User = Depends(auth.get_current_user)):
    return {
        "current_threshold": detector.base_threshold,
        "active_policy": getattr(detector, "current_policy", "p95"),
        "available_policies": detector.threshold_policies
    }

@app.get("/api/metrics")
async def get_system_metrics(current_user: models.User = Depends(auth.get_current_user)):
    eval_path = os.path.join(_BASE_DIR, "..", "..", "eval_results.json")
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            return json.load(f)
    return {"message": "Metrics not available"}

@app.post("/api/upload_dataset")
async def upload_dataset(
    file: UploadFile = File(...), 
    current_user: models.User = Depends(auth.get_current_user)
):
    if not file.filename.endswith(('.csv', '.parquet', '.txt', '.pcap')):
        raise HTTPException(status_code=400, detail="Only .csv, .parquet, or .pcap files are supported")
        
    content = await file.read()
    
    if file.filename.endswith('.pcap'):
        def process_pcap():
            from scapy.all import rdpcap
            packets = rdpcap(io.BytesIO(content))
            for pkt in packets:
                try:
                    packet_queue.put_nowait(pkt)
                    time.sleep(0.01)
                except queue.Full:
                    pass
        threading.Thread(target=process_pcap, daemon=True).start()
        return {"status": "success", "message": "PCAP processing started. Packets are being pushed to the queue."}
        
    else: # CSV / Parquet NetFlow format
        try:
            if file.filename.endswith('.parquet'):
                df = pd.read_parquet(io.BytesIO(content))
            else:
                try:
                    df = pd.read_csv(io.BytesIO(content))
                except Exception:
                    df = pd.read_csv(io.BytesIO(content), header=None)
            
            def process_netflow_dataset():
                for index, row in df.iterrows():
                    row_df = pd.DataFrame([row])
                    try:
                        scaled_feat = extractor.adapter.transform(row_df)
                        prediction = detector.predict(scaled_feat, explain=True)
                        is_anomaly = bool(prediction[0])
                        score = float(prediction[1])
                        top_features = prediction[2] if len(prediction) == 3 else []
                    except Exception as e:
                        continue

                    src = str(row.get("IPV4_SRC_ADDR", f"Flow_{index}"))
                    dst = str(row.get("IPV4_DST_ADDR", "Target_Host"))
                    proto_val = row.get("PROTOCOL", 6)
                    proto_str = "TCP" if proto_val == 6 else ("UDP" if proto_val == 17 else ("ICMP" if proto_val == 1 else str(proto_val)))
                    size = int(row.get("IN_BYTES", 0))

                    packet_info = {
                        "src": src,
                        "dst": dst,
                        "proto": proto_str,
                        "size": size,
                        "sport": int(row.get("L4_SRC_PORT", 0)),
                        "dport": int(row.get("L4_DST_PORT", 0)),
                        "in_bytes": size,
                        "out_bytes": int(row.get("OUT_BYTES", 0)),
                        "in_pkts": int(row.get("IN_PKTS", 1)),
                        "out_pkts": int(row.get("OUT_PKTS", 0)),
                        "duration": int(row.get("FLOW_DURATION_MILLISECONDS", 0)),
                        "duration_ms": int(row.get("FLOW_DURATION_MILLISECONDS", 0)),
                        "flows_10s": int(row.get("flows_10s", 1)),
                        "unique_ports": int(row.get("unique_ports_10s", 1)),
                        "flags": int(row.get("TCP_FLAGS", row.get("CLIENT_TCP_FLAGS", 0))),
                        "score": float(score),
                    }

                    attack_label = str(row.get("Attack", row.get("Label", "Normal")))
                    packet_info["attack_label"] = attack_label
                    explanation_text = f"Dataset Label: {attack_label}. Predicted: {'Anomaly' if is_anomaly else 'Normal'} (Score: {score:.4f})"
                    attack_name = "Normal Traffic"
                    if is_anomaly:
                        _, attack_name = rag_service.classify(packet_info, top_features)
                        rag_explanation = rag_service.explain_anomaly_sync(packet_info, score, top_features)
                        explanation_text = f"{explanation_text}\n\n{rag_explanation}"

                    data = {
                        "type": "log",
                        "is_anomaly": bool(is_anomaly),
                        "score": float(score),
                        "packet": packet_info,
                        "explanation": explanation_text,
                        "attack_type": attack_name,
                        "top_features": top_features if is_anomaly else []
                    }

                    if connected_clients:
                        message = json.dumps(data)
                        target_loop = main_loop
                        if target_loop is None:
                            try:
                                target_loop = asyncio.get_running_loop()
                            except RuntimeError:
                                pass

                        if target_loop and target_loop.is_running():
                            async def broadcast():
                                tasks = [client.send_text(message) for client in list(connected_clients)]
                                if tasks:
                                    await asyncio.gather(*tasks, return_exceptions=True)
                            
                            asyncio.run_coroutine_threadsafe(broadcast(), target_loop)

                    time.sleep(0.05) # stream smoothly

            threading.Thread(target=process_netflow_dataset, daemon=True).start()
            return {"status": "success", "message": f"NetFlow dataset processing started ({len(df):,} flows)."}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

main_loop = None

@app.websocket("/ws/monitor")
async def websocket_endpoint(websocket: WebSocket):
    print("New WebSocket client connected")
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            text = await websocket.receive_text() # Keep connection alive & handle heartbeats
            try:
                msg = json.loads(text)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except Exception:
                pass
    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
    except Exception:
        if websocket in connected_clients:
            connected_clients.remove(websocket)

def packet_callback(packet):
    # This runs in the high-frequency sniffing thread.
    # Put the packet into a queue and return immediately.
    try:
        if hasattr(packet, "haslayer") and packet.haslayer("IP"):
            ip_layer = packet["IP"]
            src = str(ip_layer.src)
            dst = str(ip_layer.dst)
            if src.startswith("127.") or dst.startswith("127.") or src in ("::1", "localhost") or dst in ("::1", "localhost"):
                return
            if packet.haslayer("TCP"):
                sp = int(packet["TCP"].sport)
                dp = int(packet["TCP"].dport)
                if sp in (8000, 5173, 3000) or dp in (8000, 5173, 3000):
                    return
        packet_queue.put_nowait(packet)
    except queue.Full:
        pass # Drop packet if queue is full to prevent blocking

async def process_packets():
    """Background task to process packets from the queue without blocking the sniffer or event loop."""
    while True:
        try:
            # Get packet from queue
            if not packet_queue.empty():
                packet = packet_queue.get()
                
                # Perform feature extraction and detection in worker thread to prevent event-loop starvation
                res = await asyncio.to_thread(extractor.packet_to_features, packet, return_raw=True)
                if res is not None and res[0] is not None:
                    scaled_feat, raw_data = res
                    prediction = await asyncio.to_thread(detector.predict, scaled_feat, explain=True)
                    is_anomaly = bool(prediction[0])
                    score = float(prediction[1])
                    top_features = prediction[2] if len(prediction) == 3 else []
                    
                    src_ip = raw_data.get("src", "Unknown")
                    dst_ip = raw_data.get("dst", "Unknown")
                    proto_str = raw_data.get("proto", "TCP")
                    
                    packet_info = {
                        "src": str(src_ip),
                        "dst": str(dst_ip),
                        "proto": str(proto_str),
                        "size": int(len(packet)),
                        "sport": int(raw_data.get("sport", 0)),
                        "dport": int(raw_data.get("dport", 0)),
                        "in_bytes": int(raw_data.get("in_bytes", len(packet))),
                        "out_bytes": int(raw_data.get("out_bytes", 0)),
                        "in_pkts": int(raw_data.get("in_pkts", 1)),
                        "out_pkts": int(raw_data.get("out_pkts", 0)),
                        "duration": int(raw_data.get("duration", 0)),
                        "duration_ms": int(raw_data.get("duration", 0)),
                        "flows_10s": int(raw_data.get("flows_10s", 1)),
                        "unique_ports": int(raw_data.get("unique_ports", 1)),
                        "flags": int(raw_data.get("flags", 0)),
                        "score": float(score),
                    }
                    
                    data = {
                        "type": "log",
                        "is_anomaly": bool(is_anomaly),
                        "score": float(score),
                        "packet": packet_info
                    }
                    
                    if is_anomaly:
                        _, attack_name = rag_service.classify(packet_info, top_features)
                        explanation = await asyncio.to_thread(
                            rag_service.explain_anomaly_sync, packet_info, score, top_features
                        )
                        data["explanation"] = str(explanation)
                        data["attack_type"] = str(attack_name)
                        data["top_features"] = top_features if top_features else []
                    else:
                        data["attack_type"] = "Normal Traffic"
                        data["top_features"] = []
                        data["explanation"] = ""
                        
                    # Broadcast to all connected clients safely
                    if connected_clients:
                        message = json.dumps(data)
                        dead_clients = []
                        for client in list(connected_clients):
                            try:
                                await client.send_text(message)
                            except Exception:
                                dead_clients.append(client)
                        for dc in dead_clients:
                            if dc in connected_clients:
                                connected_clients.remove(dc)
                
                packet_queue.task_done()
            else:
                await asyncio.sleep(0.005) # Yield briefly if queue is empty
        except Exception as e:
            import traceback
            print(f"Error in packet processing task: {e}")
            traceback.print_exc()
            await asyncio.sleep(0.1)

@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()
    
    # Start sniffing in a background thread
    sniffer_thread = threading.Thread(target=start_sniffing, args=(packet_callback,), daemon=True)
    sniffer_thread.start()
    
    # Start the async processing task
    asyncio.create_task(process_packets())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
