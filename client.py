# client.py

import streamlit as st
import requests
from PIL import Image
import cv2
import numpy as np
import os

# Streamlit app title and description
st.title("Face Recognition and User Login App")
st.write("Welcome to our application for face recognition and user login.")

# Function to capture an image using the webcam
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Unable to open the camera.")
        return None
    ret, frame = cap.read()
    if not ret:
        st.error("Error: Unable to capture the image.")
        return None
    cap.release()
    return frame

# Function to upload image and display it
def upload_image(label):
    uploaded_file = st.file_uploader(f"Choose an image for {label}...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded {label} Image", use_column_width=True)
        return uploaded_file
    return None

# Function to send image for face recognition
def recognize_faces(image_file):
    url = 'http://localhost:8000/facerecognize'  # FastAPI face recognition endpoint URL
    files = {'image': image_file}
    response = requests.post(url, files=files)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error {response.status_code}: {response.text}")
        return None

# Function to log in with user details
def login_user(data):
    url = 'http://localhost:8801/login'  # FastAPI login endpoint URL
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to connect to the API. Status code: {response.status_code}")
        return None

# Main Streamlit app logic
def main():
    # Capture or upload image section
    st.write("### Face Recognition and Image Upload")
    st.write("Choose an option below:")
    option = st.radio("Select an option:", ("Capture Image from Camera", "Upload Image"))

    if option == "Capture Image from Camera":
        if st.button("Capture Image"):
            frame = capture_image()
            if frame is not None:
                # Display captured image
                image_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_array)
                st.image(image_pil, caption="Captured Image", use_column_width=True)

                # Perform face recognition
                st.write("### Face Recognition Result")
                result = recognize_faces(frame)
                if result:
                    st.json(result)

    elif option == "Upload Image":
        uploaded_file = upload_image("Upload")
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Perform face recognition
            st.write("### Face Recognition Result")
            result = recognize_faces(uploaded_file)
            if result:
                st.json(result)

    # Login section
    st.write("### User Login")
    first_name = st.text_input("Enter First Name:")
    last_name = st.text_input("Enter Last Name:")
    if st.button("Login"):
        if first_name and last_name:
            data = {"first_name": first_name, "last_name": last_name}
            st.write("### Login Result")
            login_result = login_user(data)
            if login_result:
                st.json(login_result)

if __name__ == "__main__":
    main()
