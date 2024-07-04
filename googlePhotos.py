import streamlit as st
import cv2
import numpy as np
import face_recognition
from distances import retrieve_similar_images
import os
from PIL import Image

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Erreur: Impossible d'ouvrir la caméra.")
        return None
    ret, frame = cap.read()
    if not ret:
        st.error("Erreur: Impossible de capturer l'image.")
        return None
    cap.release()
    return frame

def load_images_from_folder(folder_path):
    image_list = []
    name_list = []

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if os.path.splitext(img_name)[1].lower() in ['.jpg', '.png', '.jpeg']:
            cur_img = cv2.imread(img_path)
            image_list.append(cur_img)
            img_name_without_ext = os.path.splitext(img_name)[0]
            name_list.append(img_name_without_ext)

    return image_list, name_list

# Fonction pour extraire les caractéristiques faciales et enregistrer les signatures
def find_encodings(img_list, name_list):
    signatures_db = []

    for img, name in zip(img_list, name_list):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_img)
        if len(face_locations) == 0:
            print(f"Aucun visage détecté dans l'image : {name}")
            continue

        # Extraire les caractéristiques faciales uniquement si des visages sont détectés
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        for face_encoding in face_encodings:
            signature_class = face_encoding.tolist() + [name]
            signatures_db.append(signature_class)

    if signatures_db:
        signatures_db = np.array(signatures_db)
        np.save('FaceSignatures_db.npy', signatures_db)
        print('Signatures enregistrées avec succès.')
    else:
        print('Aucun visage trouvé dans les images.')

def main():
    st.title("Recherche d'images similaires")
    st.write("Bienvenue dans notre application de recherche d'images similaires.")

    # Charger les images et enregistrer les signatures
    image_list, name_list = load_images_from_folder('./photos')
    find_encodings(image_list, name_list)

    # Bouton pour capturer une photo
    if st.button("Capturez votre photo"):
        frame = capture_image()
        if frame is not None:
            # Afficher l'image capturée
            image_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_array)
            st.image(image_pil, caption="Image capturée", use_column_width=True)

            # Extraire les caractéristiques faciales de l'image capturée
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_frame)
            if len(face_encodings) > 0:
                query_encoding = face_encodings[0]
                # Charger les signatures des visages connus
                signatures_db = np.load('FaceSignatures_db.npy')

                # Rechercher des images similaires
                similar_images = retrieve_similar_images(signatures_db, query_encoding, distance='euclidean', num_results=5)

                # Afficher les images similaires trouvées
                if similar_images:
                    st.write("Images similaires trouvées :")
                    images_folder_path = os.path.abspath('photos')

                    captured_image_label = similar_images[0][2]

                    similar_images_filtered = [sim_image for sim_image in similar_images if sim_image[2] == captured_image_label]

                    if similar_images_filtered:
                        for sim_image in similar_images_filtered:
                            image_name = sim_image[0]
                            image_label = sim_image[2]
                            image_path = os.path.join(images_folder_path, image_name + '.jpg')

                            if os.path.exists(image_path):
                                st.write(f"Image : {image_label}")
                                st.image(Image.open(image_path), caption=image_label, use_column_width=True)
                            else:
                                st.write(f"Erreur : Fichier image {image_name}.jpg introuvable dans le dossier {images_folder_path}.")
                    else:
                        st.write("Aucune image similaire de la même personne trouvée.")
                else:
                    st.write("Aucune image similaire trouvée.")
            else:
                st.write("Aucun visage trouvé dans l'image capturée.")

if __name__ == "__main__":
    main()
