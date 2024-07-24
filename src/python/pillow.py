import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import imageio

def read_grid(filename):
    """Lire une grille depuis un fichier texte."""
    with open(filename, 'r') as file:
        data = file.readlines()
    grid = [list(map(float, line.split())) for line in data]
    return np.array(grid)

def create_image(grid, step, output_dir):
    """Créer une image à partir d'une grille et sauvegarder dans un répertoire donné."""
    plt.imshow(grid, cmap='viridis', vmin=0, vmax=1)  # Normaliser les valeurs
    plt.colorbar()
    plt.title(f"Gray-Scott Model at Step {step}")
    plt.axis('off')  # Pas d'axes pour l'image
    img_path = os.path.join(output_dir, f"step_{step}.png")
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()

    # Redimensionner l'image pour s'assurer que toutes les images ont la même taille
    img = Image.open(img_path)
    img = img.resize((640, 640), Image.Resampling.LANCZOS)  # Choisissez la taille souhaitée
    img.save(img_path)

def create_video_from_images(image_files, output_file, fps=5):
    """Créer une vidéo à partir des images."""
    with imageio.get_writer(output_file, fps=fps) as writer:
        for img_file in image_files:
            img = imageio.imread(img_file)
            writer.append_data(img)

def create_gif_from_images(image_files, output_file, fps=5):
    """Créer un GIF à partir des images."""
    with imageio.get_writer(output_file, mode='I', duration=1/fps) as writer:
        for img_file in image_files:
            img = imageio.imread(img_file)
            writer.append_data(img)

def main():
    input_dir = './'  # Répertoire contenant les fichiers de sortie
    output_dir = './images'  # Répertoire pour les images générées
    os.makedirs(output_dir, exist_ok=True)
    
    steps = [1000, 2000, 3000, 4000, 5000]  # Liste des étapes que vous souhaitez inclure

    image_files = []

    for step in steps:
        filename = f"output_step_{step}.txt"
        if os.path.exists(filename):
            grid = read_grid(filename)
            create_image(grid, step, output_dir)
            image_files.append(os.path.join(output_dir, f"step_{step}.png"))
        else:
            print(f"File {filename} not found!")

    # Création de la vidéo et du GIF à partir des images
    create_video_from_images(image_files, "gray_scott_simulation.mp4", fps=5)
    create_gif_from_images(image_files, "gray_scott_simulation.gif", fps=5)

if __name__ == "__main__":
    main()

