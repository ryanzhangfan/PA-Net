#!/usr/bin/python

import os
import sys

cur_dir = os.getcwd()
ROOT_DIR = cur_dir[:cur_dir.find("samples")]
sys.path.append(ROOT_DIR)

from panet.config import Config


class BaseConfig(Config):
  NAME = "APOLLO"
  IMAGES_PER_GPU = 3

  # NUM_CLASSES = 1 + 80
  NUM_CLASSES = 1 + 7

  # RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512, 1024)


class TrainConfig(BaseConfig):
  NAME = "APOLLO"

  # GPU_COUNT = 4

  GPU_COUNT = 1


class TestConfig(BaseConfig):
  NAME = "TEST_APOLLO"

  IMAGES_PER_GPU = 1
  GPU_COUNT = 1
