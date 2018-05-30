#!/usr/bin/python

import os
import sys

import numpy as np

from skimage import io

cur_dir = os.getcwd()
ROOT_DIR = cur_dir[:cur_dir.find("samples")]
sys.path.append(ROOT_DIR)

from panet import utils

class_id_mapper = {
  33 : 1,
  34 : 2,
  35 : 3,
  36 : 4,
  38 : 5,
  39 : 6,
  40 : 7,
}

class ApolloDataset(utils.Dataset):
  def load_apollo(self, image_dir, image_ids, label_dir):
    self.add_class("apollo", 1, "car")
    self.add_class("apollo", 2, "motorbicycle")
    self.add_class("apollo", 3, "bicycle")
    self.add_class("apollo", 4, "person")
    self.add_class("apollo", 5, "truck")
    self.add_class("apollo", 6, "bus")
    self.add_class("apollo", 7, "tricycle")

    for i, img_id in enumerate(image_ids):
      self.add_image(
        source="apollo",
        image_id=i,
        path=os.path.join(image_dir, img_id + ".jpg"),
        name=img_id,
        label=os.path.join(label_dir, img_id + "_instanceIds.png")
      )

  def load_mask(self, image_id):
    img_info = self.image_info[image_id]
    if img_info["source"] != "apollo":
      return super(self.__class__, self).load_mask(image_id)

    mask_img = io.imread(img_info["label"])
    height, width = mask_img.shape

    instances = [ x for x in np.unique(mask_img) if x//1000 in [33, 34, 35, 36, 38, 39, 40]]

    class_ids = []

    mask = np.zeros((height, width, len(instances)), dtype=np.bool)

    for i, ins in enumerate(instances):
      class_id = class_id_mapper[int(ins//1000)]
      class_ids.append(class_id)

      mask[:, :, i] = np.where(mask_img==ins, True, False)
    
    return mask, np.array(class_ids).astype(np.int32)


if __name__ == "__main__":
  dataset = ApolloDataset()

  image_dir = "/nfs/project/CVPR_WAD2018/ch4_instance_segmentation/train/images/"
  label_dir = "/nfs/project/CVPR_WAD2018/ch4_instance_segmentation/train/labels/"

  # image_ids = ["170908_061502408_Camera_5", "170908_061502408_Camera_6"]
  image_ids = ["170908_082403474_Camera_5"]

  dataset.load_apollo(image_dir, image_ids, label_dir)

  for x in dataset.image_info:
    mask, class_ids = dataset.load_mask(x["id"])
    for i in range(len(class_ids)):
      print(class_ids[i])
      img = np.zeros((mask[:, :, i].shape[0], mask[:, :, i].shape[1], 3))
      img[:, :, 0] = img[:, :, 1] = img[:, :, 2] = np.where(mask[:, :, i], 255, 0)
      import cv2
      img = cv2.resize(img, (1024, 1024))
      cv2.imshow("1", img)
      cv2.waitKey(0)
