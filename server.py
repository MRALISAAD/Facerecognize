from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import face_recognition
import numpy as np
import os
import shutil

app = FastAPI()

# Allow all origins
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for image upload
class ImageUpload(BaseModel):
    image: UploadFile

# Function to calculate face encoding from image file
def calculate_face_encoding(file_path):
    image = face_recognition.load_image_file(file_path)
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) == 0:
        raise HTTPException(status_code=400, detail="No face found in the image")
    face_encodings = face_recognition.face_encodings(image, face_locations)
    return face_encodings[0]

# Endpoint for face recognition
@app.post('/facerecognize')
async def recognize_faces(image: UploadFile = File(...)):
    try:
        # Save the uploaded image to a temporary file
        temp_file = f"./temp/{image.filename}"
        with open(temp_file, "wb") as temp_image:
            shutil.copyfileobj(image.file, temp_image)

        # Calculate face encoding from the uploaded image
        face_encoding = calculate_face_encoding(temp_file)

        # Cleanup: delete temporary file
        os.remove(temp_file)

        # Return face encoding as response (for demonstration)
        return {"face_encoding": face_encoding.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
