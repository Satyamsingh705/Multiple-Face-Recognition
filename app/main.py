from fastapi import FastAPI, UploadFile, Form, File,Request, HTTPException
from fastapi.responses import HTMLResponse,FileResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.database import Base, engine, SessionLocal
from app.models import Person
from app.face_encoder import FaceEncoder
import numpy as np
import pickle
import uuid
import os
import cv2
# NEW: CORS
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import IntegrityError  # added

app = FastAPI()
# NEW: Allow localhost and 127.0.0.1 (same-port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://0.0.0.0:8000",
        "null"  # for file:// origin if opened directly (not recommended)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)
templates = Jinja2Templates(directory="templates")
encoder = FaceEncoder()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("train.html", {"request": request})

# Optional: serve .html paths directly so they don't 404
@app.get("/train.html", response_class=HTMLResponse)
async def train_html(request: Request):
    return templates.TemplateResponse("train.html", {"request": request})

@app.get("/recognize.html", response_class=HTMLResponse)
async def recognize_html(request: Request):
    return templates.TemplateResponse("recognize.html", {"request": request})

@app.post("/train/")
async def train_person(
    request: Request,  # NEW: detect HTML form submissions
    person_id: int = Form(...),
    name: str = Form(...),
    files: list[UploadFile] = File(...)
):
    db: Session = SessionLocal()
    try:
        wants_html = "text/html" in (request.headers.get("accept") or "").lower()  # NEW

        # Check if ID already exists -> reject with clear error
        existing = db.query(Person).filter(Person.id == person_id).first()
        if existing:
            # NEW: popup for browser form posts
            if wants_html:
                return HTMLResponse(
                    '<script>alert("Id is present. Try with new id"); history.back();</script>',
                    status_code=400
                )
            raise HTTPException(status_code=400, detail="Id is present. Try with new id")

        embeddings = []

        for file in files:
            contents = await file.read()
            emb = encoder.encode_image(contents)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            if wants_html:  # NEW
                return HTMLResponse(
                    f'<script>alert("No valid faces found for {name}"); history.back();</script>',
                    status_code=400
                )
            raise HTTPException(status_code=400, detail=f"No valid faces found for {name}")

        avg_embedding = np.mean(embeddings, axis=0)
        serialized = pickle.dumps(avg_embedding)

        person = Person(id=person_id, name=name, embedding=serialized)
        db.add(person)  # insert only (no upsert)
        db.commit()
        return {"message": f"✅ {name} (ID={person_id}) trained successfully!"}
    except IntegrityError:
        db.rollback()
        # Covers unique name or any PK violations enforced by DB
        if "text/html" in (request.headers.get("accept") or "").lower():  # NEW
            return HTMLResponse(
                '<script>alert("Id is present. Try with new id"); history.back();</script>',
                status_code=400
            )
        raise HTTPException(status_code=400, detail="Id is present. Try with new id")
    finally:
        db.close()

# Support POST without trailing slash
@app.post("/train")
async def train_person_no_slash(
    request: Request,  # NEW
    person_id: int = Form(...),
    name: str = Form(...),
    files: list[UploadFile] = File(...)
):
    return await train_person(request, person_id, name, files)

@app.get("/recognize-page", response_class=HTMLResponse)
async def recognize_page(request: Request):
    return templates.TemplateResponse("recognize.html", {"request": request})

@app.post("/recognize/")
async def recognize_faces(file: UploadFile = File(...)):
    db: Session = SessionLocal()
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Get all known embeddings from DB
    persons = db.query(Person).all()
    known_embeddings = []
    known_names = []

    for p in persons:
        emb = pickle.loads(p.embedding)
        known_embeddings.append(emb)
        known_names.append(p.name)

    known_embeddings = np.array(known_embeddings)
    db.close()

    # Detect faces in the uploaded image
    faces = encoder.app.get(img)
    threshold_cos = 0.4  # tweak for sensitivity

    for face in faces:
        emb = encoder.l2_normalize(face.embedding)
        cos_sim = np.dot(known_embeddings, emb)
        idx = np.argmax(cos_sim)
        max_sim = cos_sim[idx]

        name = "Unknown"
        if max_sim > threshold_cos:
            name = known_names[idx]

        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{name}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Save result image temporarily
    output_path = f"recognized_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(output_path, img)

    return FileResponse(output_path, media_type="image/jpeg")

# Support POST without trailing slash
@app.post("/recognize")
async def recognize_faces_no_slash(file: UploadFile = File(...)):
    return await recognize_faces(file)