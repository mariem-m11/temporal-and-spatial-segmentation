import cv2
import numpy as np
import os
# Charger la vidéo
video_path = "E:/RT5/Analyse vidéo/Vidéo_TP1.mp4"
video = cv2.VideoCapture(video_path)

# Vérifier si la vidéo s'ouvre correctement
if not video.isOpened():
    print("Erreur : Impossible de lire la vidéo")
    exit()

# Choix du paramètre alpha (entre 0 et 1)
alpha = 0.7 # Ajustez la valeur de alpha en fonction de votre vidéo

# Lire la première frame pour initialiser B(t-1)
ret, frame = video.read()
if not ret:
    print("Erreur lors de la lecture de la première frame")
    video.release()
    exit()

# Convertir la première frame en niveaux de gris pour B(t-1)
previous_background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Créer un dossier pour sauvegarder les frames segmentés
frames_folder = "segmented_frames_adaptive_bg/alpha07"
if not os.path.exists(frames_folder):
    os.makedirs(frames_folder)

frame_count = 0  # Initialisation du compteur de frames

# Boucle pour lire les frames suivantes
while True:
    ret, frame = video.read()  # Lire la frame suivante I(t)
    if not ret:
        print("Fin de la vidéo")
        break

    # Conversion en niveaux de gris de la frame actuelle I(t)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculer la différence absolue entre B(t-1) et I(t)
    diff = cv2.absdiff(previous_background, gray_frame)

    # Appliquer un seuillage pour obtenir une image binaire
    _, movement_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Afficher les résultats
    cv2.imshow('Vidéo d\'origine', frame)
    cv2.imshow('Mouvement détecté (Adaptive Background Subtraction)', movement_mask)

    # Sauvegarder une frame segmentée toutes les 30 frames
    if frame_count % 30 == 0:
        frame_filename = os.path.join(frames_folder, f'frame_adaptive_bg_{frame_count}.jpg')
        cv2.imwrite(frame_filename, movement_mask)
        print(f"Frame segmentée {frame_count} sauvegardée : {frame_filename}")

    # Mettre à jour le fond B(t) avec la formule adaptative
    previous_background = cv2.addWeighted(gray_frame, alpha, previous_background, 1 - alpha, 0)

    frame_count += 1  # Incrémenter le compteur de frames

    # Attendre 30 ms entre chaque frame, quitter si l'utilisateur appuie sur 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        print("Lecture interrompue par l'utilisateur")
        break

# Libérer la vidéo et fermer toutes les fenêtres
video.release()
cv2.destroyAllWindows()
