import cv2
import numpy as np

# Charger la vidéo
video_path = "E:/RT5/Analyse vidéo/Vidéo_TP1.mp4"
video = cv2.VideoCapture(video_path)

# Vérifier si la vidéo s'ouvre correctement
if not video.isOpened():
    print("Erreur : Impossible de lire la vidéo")
    exit()

# Choix manuel du seuil initial T
T_initial = int(input("Entrez le seuil initial T : "))

# Initialisation des variables
T = T_initial  # Initialiser T avec la valeur choisie manuellement
previous_T = 0  # Pour vérifier la convergence

# Boucle pour lire les frames
while True:
    ret, frame = video.read()  # Lire un frame
    if not ret:
        print("Fin de la vidéo")
        break

    # Conversion en niveaux de gris
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Itérer pour trouver T_final
    while abs(T - previous_T) > 1:  # Critère de convergence
        # Séparer les pixels en deux groupes
        foreground = gray_frame[gray_frame >= T]  # Pixels de l'objet (>= T)
        background = gray_frame[gray_frame < T]   # Pixels de l'arrière-plan (< T)
        
        # Calculer les moyennes des deux groupes
        if len(foreground) > 0:
            mean_foreground = np.mean(foreground)
        else:
            mean_foreground = 0  # Éviter division par zéro

        if len(background) > 0:
            mean_background = np.mean(background)
        else:
            mean_background = 0  # Éviter division par zéro

        # Mettre à jour le seuil (nouveau T)
        previous_T = T
        T = (mean_foreground + mean_background) / 2  # Calcul du nouveau seuil

    # Appliquer le seuillage avec le seuil T final
    ret, thresholded_frame = cv2.threshold(gray_frame, T, 255, cv2.THRESH_BINARY)

    # Afficher la vidéo segmentée en temps réel
    cv2.imshow('Vidéo segmentée par seuillage heuristique', thresholded_frame)

    # Attendre 50 ms entre chaque frame
    if cv2.waitKey(50) & 0xFF == ord('q'):  # Quitter si l'utilisateur appuie sur 'q'
        print("Lecture interrompue par l'utilisateur")
        break

# Afficher la valeur finale de T_final
print(f"Le seuil final T_final est : {T}")

# Libérer la vidéo et fermer toutes les fenêtres
video.release()
cv2.destroyAllWindows()
