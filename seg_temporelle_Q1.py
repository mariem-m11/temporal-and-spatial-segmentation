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

# Lire la première frame pour initialiser B(t-1)
ret, frame = video.read()
if not ret:
    print("Erreur lors de la lecture de la première frame")
    video.release()
    exit()

# Convertir la première frame en niveaux de gris pour B(t-1)
previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Créer un dossier pour sauvegarder les frames segmentés
frames_folder = "segmented_frames_temporal_t_1"
if not os.path.exists(frames_folder):
    os.makedirs(frames_folder)

frame_count = 0  # Initialisation du compteur de frames

# Boucle pour lire les frames suivantes et calculer la différence
while True:
    ret, frame = video.read()  # Lire la frame suivante
    if not ret:
        print("Fin de la vidéo")
        break

    # Conversion en niveaux de gris de la frame actuelle I(t)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculer la différence absolue entre B(t-1) et I(t)
    diff = cv2.absdiff(previous_frame, gray_frame)

    # Appliquer un seuillage sur la différence pour obtenir une image binaire (lambda = 30 ici)
    _, movement_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Afficher la frame originale et la frame résultante de la soustraction
    cv2.imshow('Vidéo d\'origine', frame)
    cv2.imshow('Vidéo segmentée par soustraction d\'images', movement_mask)

    # Sauvegarder une frame segmentée toutes les 30 frames (par exemple)
    if frame_count % 30 == 0:
        frame_filename = os.path.join(frames_folder, f'segmented_frame_{frame_count}.jpg')
        cv2.imwrite(frame_filename, movement_mask)
        print(f"Frame segmentée {frame_count} sauvegardée : {frame_filename}")

    # Mettre à jour B(t-1) avec la frame actuelle I(t) pour la prochaine itération
    previous_frame = gray_frame.copy()

    frame_count += 1  # Incrémenter le compteur de frames

    # Attendre 30 ms entre chaque frame, quitter si l'utilisateur appuie sur 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        print("Lecture interrompue par l'utilisateur")
        break

# Libérer la vidéo et fermer toutes les fenêtres
video.release()
cv2.destroyAllWindows()
