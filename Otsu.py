import cv2
import os

# Charger la vidéo
video_path = "E:/RT5/Analyse vidéo/Vidéo_TP1.mp4"
video = cv2.VideoCapture(video_path)

# Vérifier si la vidéo s'ouvre correctement
if not video.isOpened():
    print("Erreur : Impossible de lire la vidéo")
    exit()

# Créer un dossier pour sauvegarder les frames segmentés avec Otsu
frames_folder = "segmented_frames_otsu"
if not os.path.exists(frames_folder):
    os.makedirs(frames_folder)

frame_count = 0  # Initialisation du compteur de frames

# Boucle pour lire et segmenter les frames avec la méthode d'Otsu
while True:
    ret, frame = video.read()  # Lire un frame
    if not ret:
        print("Fin de la vidéo")
        break

    # Conversion en niveaux de gris
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Appliquer la méthode d'Otsu pour le seuillage automatique
    ret, otsu_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Afficher la vidéo d'origine
    cv2.imshow('Vidéo d\'origine', frame)

    # Afficher la vidéo segmentée avec la méthode d'Otsu
    cv2.imshow('Vidéo segmentée avec Otsu', otsu_frame)

    # Sauvegarder une frame toutes les 30 frames
    if frame_count % 30 == 0:  # Sauvegarder toutes les 30 frames segmentés
        frame_filename = os.path.join(frames_folder, f'otsu_frame_{frame_count}.jpg')
        cv2.imwrite(frame_filename, otsu_frame)
        print(f"Frame segmentée avec Otsu {frame_count} sauvegardée : {frame_filename}")

    frame_count += 1  # Incrémenter le compteur de frames

    # Attendre 30 ms entre chaque frame
    if cv2.waitKey(30) & 0xFF == ord('q'):  # Quitter si l'utilisateur appuie sur 'q'
        print("Lecture interrompue par l'utilisateur")
        break

# Libérer la vidéo et fermer toutes les fenêtres
video.release()
cv2.destroyAllWindows()
