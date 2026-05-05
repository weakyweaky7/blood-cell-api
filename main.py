from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from hybrid_model import HybridModel

app = FastAPI(
    title="Blood Cell Classifier API",
    description="EfficientNet + VAE Hybrid Model",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model...")
model = HybridModel()
print("Model loaded!")

@app.get("/")
def root():
    return {"message": "ML API is running", "model": "EfficientNet + VAE Hybrid"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result = model.predict(image)
    result["filename"] = file.filename
    return result
