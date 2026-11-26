# Facerecognize — reconnaissance faciale (FastAPI + Streamlit)

Application de connexion par reconnaissance faciale : un client Streamlit capture une image via la webcam et l'envoie à une API FastAPI qui compare l'encodage du visage à une base de signatures connues.

## Fonctionnement

1. **Client** (`client.py`, Streamlit) : capture une photo avec la webcam (OpenCV) et l'envoie à l'API.
2. **API** (`server.py`, FastAPI) : endpoint `POST /facerecognize` qui calcule l'encodage du visage (`face_recognition`) et le compare aux signatures enregistrées.
3. **Signatures** (`signatures.py`, `descriptors.py`, `distances.py`) : génération des encodages à partir de photos de référence et calcul des distances.

## Stack technique

- Python, FastAPI, Uvicorn
- Streamlit, OpenCV, Pillow
- face_recognition (dlib), NumPy
- Déploiement : GitHub Actions → Azure Web App

## Démarrage

```bash
pip install -r requirements.txt

# 1. Lancer l'API
python server.py            # http://localhost:8000

# 2. Lancer le client
streamlit run client.py
```

## Auteur

Ali Saad — [github.com/MRALISAAD](https://github.com/MRALISAAD)
