import numpy as np
from skimage import io, color, filters, exposure
from skimage.transform import resize
import matplotlib.pyplot as plt
import argparse
import os

# Converting the original image to grayscale
def preprocess_image(image, puzzle_size=(20, 20)):    
    if len(image.shape) == 3:
        image = color.rgb2gray(image)
    
    
    image = exposure.equalize_hist(image) # Enhancing the contrast
    
    
    resized_image = resize(image, puzzle_size, anti_aliasing=True) # Resizing the image to the puzzle size
    
    return resized_image

# applying thresholding to convert the image to binary
def apply_threshold(image, threshold_method='mean'):
    
    if threshold_method == 'otsu':
        threshold = filters.threshold_otsu(image)
    elif threshold_method == 'mean':
        threshold = np.mean(image)
    else:
        raise ValueError("Please Use 'otsu' or 'mean'.")
    
    
    binary_image = (image > threshold).astype(int) # Convert to a strict binary image (0 and 1)
    
    return binary_image
# Generating row and column clues for the nonogram.
def generate_clues(binary_image):
    
    row_clues = []
    col_clues = []
    
    for row in binary_image:
        clue = []
        count = 0
        for pixel in row:
            if pixel:
                count += 1
            elif count > 0:
                clue.append(count)
                count = 0
        if count > 0:
            clue.append(count)
        row_clues.append(clue if clue else [0])
    
    for col in binary_image.T:
        clue = []
        count = 0
        for pixel in col:
            if pixel:
                count += 1
            elif count > 0:
                clue.append(count)
                count = 0
        if count > 0:
            clue.append(count)
        col_clues.append(clue if clue else [0])
    
    return row_clues, col_clues

# Creating the nonogram from the image
def create_nonogram(image_path, puzzle_size=(20, 20), threshold_method='mean'):
    
    
    image = io.imread(image_path) # Loading the image
    
    preprocessed_image = preprocess_image(image, puzzle_size)
    
    
    binary_image = apply_threshold(preprocessed_image, threshold_method) # Apply thresholding
    
    # Generate nonogram clues
    row_clues, col_clues = generate_clues(binary_image)
    
    return binary_image, row_clues, col_clues
#Visualizing the nonogram and saving the solved puzzle
def visualize_and_save_nonogram(binary_image, row_clues, col_clues, save_path):
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    
    ax.imshow(binary_image, cmap='gray', vmin=0, vmax=1) #  Ploting the binary image in strict black-and-white

    for i, row_values in enumerate(row_clues):
        row_text = ' '.join(map(str, row_values))
        ax.text(-1.5, i, row_text, va='center', ha='right', fontsize=4, color='black')

    for j, col_values in enumerate(col_clues):
        col_text = ' '.join(map(str, col_values))
        ax.text(j, -1.5, col_text, va='bottom', ha='center', rotation=90, fontsize=4, color='black')
    
    # Adding grid
    ax.set_xticks(np.arange(binary_image.shape[1]) - 0.5, minor=True)
    ax.set_yticks(np.arange(binary_image.shape[0]) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    
    # Removing tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Saving the plot as an image file
    plt.savefig(f"{save_path}/puzzle image {image_name}.png")
    plt.show()

    # Saving the row and column clues to a text file
    with open(f"{save_path}/text puzzle {image_name}.txt", "w") as f:
        f.write("Row Clues:\n")
        for row in row_clues:
            f.write(' '.join(map(str, row)) + '\n')
        f.write("\nColumn Clues:\n")
        for col in col_clues:
            f.write(' '.join(map(str, col)) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a nonogram from an image")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    args = parser.parse_args()
    image_path = args.image_path
    image_name=str.split(image_path,'\\')[-1].split('.')[0].split(' ')[-1]
    save_path= os.getcwd()
    puzzle_size = (120, 100)  # Adjust size as needed for the Nonogram grid
    binary_image, row_clues, col_clues = create_nonogram(image_path, puzzle_size)
    visualize_and_save_nonogram(binary_image, row_clues, col_clues, save_path)
