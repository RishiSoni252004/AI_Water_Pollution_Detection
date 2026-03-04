from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from backend.inference import predict_image, load_model
import os
import shutil

app = FastAPI(title="Water Pollution Detection API")

# Ensure directories exist
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("model", exist_ok=True)

# Mount static files (CSS, JS, Images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates (HTML files)
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
async def startup_event():
    # Load the model on startup
    load_model(model_path='model/water_pollution_model.pth', classes_path='model/classes.txt')

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        return {"error": "File must be an image."}

    try:
        # Save image for display in frontend
        file_location = f"static/uploads/{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Read image to memory for inference
        with open(file_location, "rb") as f:
            image_bytes = f.read()
            
        # Get prediction
        result = predict_image(image_bytes)
        
        # Add image path to result
        result["image_url"] = f"/{file_location}"
        return result

    except Exception as e:
        return {"error": str(e)}
