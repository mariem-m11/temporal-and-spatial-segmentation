import cv2
import os
import numpy as np

# Charger la vidéo
video_path = "E:/RT5/Analyse vidéo/Vidéo_TP1.mp4"
video = cv2.VideoCapture(video_path)

# Vérifier si la vidéo s'ouvre correctement
if not video.isOpened():
    print("Erreur : Impossible de lire la vidéo")
    exit()

# Obtenir le framerate (FPS) de la vidéo
fps = video.get(cv2.CAP_PROP_FPS)
print(f"Framerate de la vidéo : {fps} FPS")

# Choix de N (décalage en frames)
N = 20 

# Créer un dossier pour sauvegarder les frames segmentés
frames_folder = "segmented_frames_3frame_diff"
if not os.path.exists(frames_folder):
    os.makedirs(frames_folder)

frame_buffer = []  # Buffer pour stocker les frames afin d'accéder à I(t+N)
frame_count = 0  # Pour garder la trace des frames

# Lire les premiers N frames et les stocker dans le buffer
for i in range(N):
    ret, frame = video.read()
    if not ret:
        print("Erreur lors de la lecture des frames")
        video.release()
        exit()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_buffer.append(gray_frame)

# Boucle pour lire les frames suivantes et calculer la différence
while True:
    ret, current_frame = video.read()  # Lire la frame suivante I(t)
    if not ret:
        print("Fin de la vidéo")
        break

    # Convertir la frame actuelle en niveaux de gris
    gray_current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Lire le frame I(t + N) depuis le buffer
    future_frame = frame_buffer.pop(0)

    # Calculer D(N) = || I(t) - I(t + N) || pour détecter les mouvements
    movement_mask = cv2.absdiff(gray_current_frame, future_frame)

    # Appliquer un seuillage pour éliminer les petits changements et ne conserver que les mouvements significatifs
    _, thresholded_movement = cv2.threshold(movement_mask, 30, 255, cv2.THRESH_BINARY)

    # Afficher les résultats
    cv2.imshow('Vidéo d\'origine', current_frame)
    cv2.imshow('Mouvement détecté (différence de frames)', thresholded_movement)

    # Sauvegarder une frame segmentée toutes les 30 frames
    if frame_count % 30 == 0:
        frame_filename = os.path.join(frames_folder, f'frame_movement_{frame_count}.jpg')
        cv2.imwrite(frame_filename, thresholded_movement)
        print(f"Frame segmentée {frame_count} sauvegardée : {frame_filename}")

    # Ajouter la nouvelle frame au buffer
    frame_buffer.append(gray_current_frame)

    frame_count += 1  # Incrémenter le compteur de frames

    # Attendre 30 ms entre chaque frame, quitter si l'utilisateur appuie sur 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        print("Lecture interrompue par l'utilisateur")
        break

# Libérer la vidéo et fermer toutes les fenêtres
video.release()
cv2.destroyAllWindows()
