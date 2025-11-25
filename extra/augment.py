import numpy as np
from PIL import Image
from pathlib import Path
import sys
cwd = Path.cwd()
sys.path.append(cwd.as_posix())
sys.path.append((cwd / 'test').as_posix())
from extra.datasets import fetch_mnist
from tqdm import trange

def augment_img(X, rotate=10, px=3):
  import random
  
  def add_fog(image_array, density=0.1, fog_color=(192, 192, 192)):
    """
    Applies a simple fog effect to a given image (NumPy array).
  
    Args:
      image_array (np.ndarray): The input image as a NumPy array (H, W) or (H, W, C).
      density (float): The density of the fog, between 0 and 1. Higher values mean denser fog.
      fog_color (tuple): RGB tuple for the fog color (e.g., (192, 192, 192) for light grey).
  
    Returns:
      np.ndarray: The image with the fog effect applied.
    """
    if image_array.ndim == 2:  # Grayscale image
      # Convert to 3 channels for consistent fog application, then back to grayscale
      temp_image = np.stack([image_array, image_array, image_array], axis=-1)
      has_alpha = False
    elif image_array.shape[-1] == 4: # RGBA image
      temp_image = image_array[..., :3] # Only apply fog to RGB channels
      has_alpha = True
    else: # RGB image
      temp_image = image_array
      has_alpha = False
  
    # Create a fog layer with the specified color and density
    fog_layer = np.full(temp_image.shape, fog_color, dtype=np.uint8)
  
    # Blend the fog layer with the image
    # new_pixel = (1 - density) * original_pixel + density * fog_color
    foggy_image = ((1 - density) * temp_image + density * fog_layer).astype(np.uint8)
  
    if image_array.ndim == 2:
      # Convert back to grayscale if original was grayscale
      foggy_image = np.dot(foggy_image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    elif has_alpha:
      # Re-add alpha channel if it existed
      alpha_channel = image_array[..., 3:]
      foggy_image = np.concatenate((foggy_image, alpha_channel), axis=-1)
  
    return foggy_image
  
  def augment_img(X, rotate=10, px=3):
    Xaug = np.zeros_like(X)
    for i in trange(len(X)):
      im = Image.fromarray(X[i])
      im = im.rotate(np.random.randint(-rotate,rotate), resample=Image.BICUBIC)
      w, h = X.shape[1:]
      #upper left, lower left, lower right, upper right
      quad = np.random.randint(-px,px,size=(8)) + np.array([0,0,0,h,w,h,w,0])
      im = im.transform((w, h), Image.QUAD, quad, resample=Image.BICUBIC)
  
      # Apply fog augmentation with a 20% probability
      if random.random() < 0.2:
        # Convert PIL Image back to numpy array for add_fog
        im_array = np.array(im)
        im_array = add_fog(im_array, density=np.random.uniform(0.05, 0.3), fog_color=(np.random.randint(150, 220), np.random.randint(150, 220), np.random.randint(150, 220)))
        # Convert back to PIL Image for consistency if further PIL operations are expected, though none are here.
        im = Image.fromarray(im_array)
      Xaug[i] = im
    return Xaug

if __name__ == "__main__":
  import matplotlib.pyplot as plt
  X_train, Y_train, X_test, Y_test = fetch_mnist()
  X_train = X_train.reshape(-1, 28, 28).astype(np.uint8)
  X_test = X_test.reshape(-1, 28, 28).astype(np.uint8)
  X = np.vstack([X_train[:1]]*10+[X_train[1:2]]*10)
  fig, a = plt.subplots(2,len(X))
  Xaug = augment_img(X)
  for i in range(len(X)):
    a[0][i].imshow(X[i], cmap='gray')
    a[1][i].imshow(Xaug[i],cmap='gray')
    a[0][i].axis('off')
    a[1][i].axis('off')
  plt.show()

  #create some nice gifs for doc?!
  for i in range(10):
    im = Image.fromarray(X_train[7353+i])
    im_aug = [Image.fromarray(x) for x in augment_img(np.array([X_train[7353+i]]*100))]
    im.save(f"aug{i}.gif", save_all=True, append_images=im_aug, duration=100, loop=0)
