#!/usr/bin/python

import datetime
import os
import sys

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s %(levelname)s: %(message)s')

cur_dir = os.getcwd()
ROOT_DIR = cur_dir[:cur_dir.find("samples")]
sys.path.append(ROOT_DIR)

from config import TrainConfig, TestConfig
from dataset import ApolloDataset

import panet.model as modellib
import imgaug

# we use pretrained COCO model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")

LOGS_DIR = os.path.join(ROOT_DIR, "logs")

IMAGE_DIR = "/nfs/project/CVPR_WAD2018/ch4_instance_segmentation/train/images/"
LABEL_DIR = "/nfs/project/CVPR_WAD2018/ch4_instance_segmentation/train/labels/"
TEST_DIR  = "/nfs/project/CVPR_WAD2018/ch4_instance_segmentation/test/"


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Train Mask R-CNN on Apollo.')

  parser.add_argument("command", metavar="<command>", help="'train' or 'evaluate' or 'trial'")
  parser.add_argument(
      "--model",
      metavar="/path/to/weights.h5",
      help="path to pretrained model",
      default=None,
    )
  parser.add_argument(
      "--logs",
      metavar="/path/to/save/logs/and/models",
      help="path to save logs and models",
      default=LOGS_DIR,
    )
  # train
  parser.add_argument(
      "--image",
      metavar="/path/to/train/images",
      help="path to images",
      default=IMAGE_DIR,
    )
  parser.add_argument(
      "--label",
      metavar="/path/to/labels",
      help="path to labels(only for 'train')",
      default=LABEL_DIR,
    )
  parser.add_argument(
      "--valid",
      metavar="/path/to/valid/img/list/file",
      help="path to valid images list file",
      default="/nfs/project/CVPR_WAD2018/ch4_instance_segmentation/filtered_valid.txt",
    )

  # evaluate
  parser.add_argument(
      "--test",
      metavar="/path/to/test/images",
      help="path to test images",
      default=TEST_DIR,
    )
  parser.add_argument(
      "--tvalid",
      metavar="/path/to/test/images/file/list",
      help="path to test images file list",
      default="/nfs/project/CVPR_WAD2018/ch4_instance_segmentation/test.txt",
    )
  parser.add_argument(
      "--mask",
      metavar="/path/to/save/predict/mask",
      help="path to save predict mask",
      default=ROOT_DIR+"/logs/masks/",
    )
  parser.add_argument(
      "--save_mask",
      metavar="True or False",
      help="whether save mask or not",
      default=False,
    )
  parser.add_argument(
      "--eval_model",
      metavar="modeltype",
      help="evaluation model type, coco or apollo",
      default="apollo",
    )
  parser.add_argument(
      "--submit",
      metavar="/path/to/result/file",
      help="path to result file",
      default=ROOT_DIR+"/samples/apollo/submit/res_"+datetime.datetime.now().strftime("%Y%m%dT%H%M")+".csv",
    )

  args = parser.parse_args()

  # config
  if args.command == "train" or args.command == "trial":
    config = TrainConfig()
    if args.command == "trial":
      config.STEPS_PER_EPOCH = 10
  else:
    config = TestConfig()
  config.display()

  # model
  if args.command == "train" or args.command == "trial":
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
  else:
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

  if args.model:
    if args.model.lower() == "coco_backbone" or args.model.lower() == "coco":
      model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
      model_path = model.find_last()[1]
    elif args.model.lower() == "imagenet":
      model_path = model.get_imagenet_werights()
    else:
      model_path = args.model

    logging.info("Loading weights " +  model_path)
    if args.model.lower() == "coco_backbone":
      model.load_weights(model_path, by_name=True, exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask', "mrcnn_class_conv1"])
    else:
      model.load_weights(model_path, by_name=True)
  
  # data & label
  if args.command == "train" or args.command == "trial":
    f = open(args.valid)
    lines = f.readlines()
    f.close()

    import random
    samples = [ x.strip() for x in lines ]
    random.shuffle(samples)

    if args.command == "train":
      train = samples[:-1000]
      val = samples[-1000:]
    elif args.command == "trial":
      train = samples[:20]
      val = samples[20:30]
  else:
    f = open(args.tvalid)
    lines = f.readlines()
    f.close()
    test = [ x.strip() for x in lines ]

  # train or evaluate
  if args.command == "train":
    # save train and validate image ids
    os.system("mkdir " + model.log_dir)
    train_file = open(os.path.join(model.log_dir, "train_ids.txt"), 'w+')
    for each_id in train:
      print(each_id, file=train_file)
    train_file.close()

    val_file = open(os.path.join(model.log_dir, "val_ids.txt"), 'w+')
    for each_id in val:
      print(each_id, file=val_file)
    val_file.close()

    # cook data
    dataset_train = ApolloDataset()
    dataset_train.load_apollo(args.image, train, args.label)
    dataset_train.prepare()
    dataset_val = ApolloDataset()
    dataset_val.load_apollo(args.image, val, args.label)
    dataset_val.prepare()

    augmentation = imgaug.augmenters.Fliplr(0.5)

    # train
    logging.info("Training network heads")
    model.train(
      dataset_train,
      dataset_val,
      learning_rate=config.LEARNING_RATE,
      epochs=40,
      layers='heads',
      augmentation=augmentation
    )

    logging.info("Fine tune resnet stage 4 and up")
    model.train(
      dataset_train,
      dataset_val,
      learning_rate=config.LEARNING_RATE,
      epochs=120,
      layers='4+',
      augmentation=augmentation
    )

    logging.info("Fine tune all layers")
    model.train(
      dataset_train,
      dataset_val,
      learning_rate=config.LEARNING_RATE,
      epochs=160,
      layers='all',
      augmentation=augmentation
    )
  elif args.command == "trial":
    dataset_train = ApolloDataset()
    dataset_train.load_apollo(args.image, train, args.label)
    dataset_train.prepare()
    dataset_val = ApolloDataset()
    dataset_val.load_apollo(args.image, val, args.label)
    dataset_val.prepare()

    augmentation = imgaug.augmenters.Fliplr(0.5)

    model.train(
      dataset_train,
      dataset_val,
      learning_rate=config.LEARNING_RATE,
      epochs=3,
      layers='all',
      augmentation=augmentation
    )
  else:
    from utils import to_submit_format, combine_masks, remove_overlap
    from skimage import io
    submit = open(args.submit, 'w+')
    print("ImageId,LabelId,Confidence,PixelCount,EncodedPixels", file=submit)

    if args.save_mask:
      logging.info("the set is saving masks...")

    for cnt, _ in enumerate(test):

      img = io.imread(os.path.join(args.test, _ + ".jpg"))
      images = [img]

      result = model.detect(images, verbose=0)[0]

      # result = remove_overlap(result)
      res_list = to_submit_format(_, result, args.eval_model)
      for x in res_list:
        print(x, file=submit)

      # save predict masks
      if args.save_mask:
        mask_image = combine_masks(result, args.eval_model)
        io.imsave(os.path.join(args.mask, _ + "_predictIds.png"), mask_image)

      logging.info(("have completed %d images..." % (cnt+1)))
    
