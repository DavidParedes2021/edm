import argparse
import os, glob
import numpy as np
import cv2
import random

#For problems with cv2
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Adding arguments:
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoints", type=str, default="checkpoints/", help="saving checkpoints path")
opt = parser.parse_args()
print(opt)