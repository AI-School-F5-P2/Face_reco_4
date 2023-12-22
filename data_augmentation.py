# Description: This script is used to augment the data of the positive images and anchors.
import tensorflow as tf
import numpy as np
import cv2
import os
from dotenv import load_dotenv
import uuid   # Universally Unique Identifier


# Data augmentation function
def augm_data (img):
  data= []
  for i in range(7):
    img = tf.image.stateless_random_brightness(img, max_delta=0.05, seed=(1,2)) #applies random brightness adjustment to the image using
    img = tf.image.stateless_random_contrast(img, lower=0.8, upper=1, seed=(1,3)) #applies random contrast adjustment to the image using
    img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100), np.random.randint(100)))  #Random left-right flipping is applied using
    img = tf.image.stateless_random_jpeg_quality(img, 80, 100, seed=(np.random.randint(100), np.random.randint(100))) #Random adjustment of JPEG image quality
    img = tf.image.stateless_random_saturation(img, 0.5, 1, seed=(np.random.randint(100), np.random.randint(100))) #Random saturation adjustment is applied using
    data.append(img)

  return data

# loading of .env variables for the path
load_dotenv()

# Aumentation of positives 👍 and anchors ⚓ 🖼
# 👍 🖼
Pos_Path = os.getenv('Pos_Path')
Aum_Pos_Path = os.getenv('Pos_Path')


for file_name in os.listdir(os.path.join(Pos_Path)):
  img_path = os.path.join(Pos_Path, file_name)
  img = cv2.imread(img_path)
  if img is None:
    print(f"Error: Unable to read image {img_path}")
    continue  # Skip to the next iteration
  aug_imgs = augm_data(img)
  # Loop through the augmented images
  #   # Generate a unique filename using uuid
  #   # Construct the full path by joining Aum_Pos_Path and the unique filename
  #   # Write the image to the full path
  for image in aug_imgs:
    cv2.imwrite(os.path.join(Aum_Pos_Path, '{}.jpg'.format(str(uuid.uuid1()))), image.numpy())


# ⚓ 🖼
Anc_Path = os.getenv('Anc_Path')
Aum_Anc_Path = os.getenv('Anc_Path')

for file_name in os.listdir(os.path.join(Anc_Path)):
  img_path = os.path.join(Anc_Path, file_name)
  img = cv2.imread(img_path)
  if img is None:
    print(f"Error: Unable to read image {img_path}")
    continue  # Skip to the next iteration
  aug_imgs = augm_data(img)

  for image in aug_imgs:
    cv2.imwrite(os.path.join(Aum_Anc_Path, '{}.jpg'.format(str(uuid.uuid1()))), image.numpy())
