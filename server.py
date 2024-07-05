from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import face_recognition
import numpy as np
import tempfile
import os

app = FastAPI()

def calculate_face_encoding(file_path):
    image = face_recognition.load_image_file(file_path)
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) == 0:
        raise HTTPException(status_code=400, detail="No face found in the image")
    face_encodings = face_recognition.face_encodings(image, face_locations)
    return face_encodings[0]

@app.post("/api/Facerecognize")
async def Facerecognize(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file1:
            temp_file1.write(await file1.read())
            temp_file1_path = temp_file1.name

        with tempfile.NamedTemporaryFile(delete=False) as temp_file2:
            temp_file2.write(await file2.read())
            temp_file2_path = temp_file2.name

        encoding1 = calculate_face_encoding(temp_file1_path)
        encoding2 = calculate_face_encoding(temp_file2_path)

        match = face_recognition.Facerecognize([encoding1], encoding2)[0]

        # Clean up temporary files
        os.remove(temp_file1_path)
        os.remove(temp_file2_path)

        return {"match": match}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
