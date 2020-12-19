import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler


def compute_mean_and_std(dir_name: str) -> Tuple[np.ndarray, np.array]:
  '''
  Compute the mean and the standard deviation of the dataset.

  Note: convert the image in grayscale and then scale to [0,1] before computing
  mean and standard deviation

  Hints: use StandardScalar (check import statement)

  Args:
  -   dir_name: the path of the root dir
  Returns:
  -   mean: mean value of the dataset (np.array containing a scalar value)
  -   std: standard deviation of th dataset (np.array containing a scalar value)
  '''

  mean = None
  std = None

  ############################################################################
  # Student code begin
  ############################################################################
  scaler = StandardScaler()
  # initialize dataset
  dataset = []

  # parse through both files in data directory 
  for f in os.listdir(dir_name):
    # path creation 
    f = os.path.join(dir_name, f)
    for f_sub in os.listdir(f):
      f_sub = os.path.join(f, f_sub)
      for img in os.listdir(f_sub):
        # get image at sub-path
        img = os.path.join(f_sub, img)
        # reshape
        i = Image.Image.split(Image.open(img))
        i = np.reshape(np.array(i[0])/255, (-1, 1))
        scaler.partial_fit(i)
        
    # use standardScalar for mean and variance 
    mean = scaler.mean_
    std = np.sqrt(scaler.var_)
  

  ############################################################################
  # Student code end
  ############################################################################
  return mean, std
