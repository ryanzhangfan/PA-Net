import copy
import numpy as np
import multiprocessing

my_class = {
    "car": 33,
    "motorcycle": 34,
    "bicycle": 35,
    "person": 36,
    "truck": 38,
    "bus": 39,
    "tricycle": 40,
  }


my_class_ids = [ 33, 34, 35, 36, 38, 39, 40 ]


coco_class = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
              'bus', 'train', 'truck', 'boat', 'traffic light',
              'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
              'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
              'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
              'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
              'kite', 'baseball bat', 'baseball glove', 'skateboard',
              'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
              'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
              'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
              'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
              'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
              'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
              'teddy bear', 'hair drier', 'toothbrush']

apollo_class = ['BG', 'car', 'motorcycle', 'bicycle', 'person', 'truck', 'bus', 'tricycle']


def get_coor(index):
  return int(index / 3384), index % 3384


def sort_by_scores(result):
  x = copy.deepcopy(result)
  indexs = np.argsort(-x['scores'])

  for k in x.keys():
    x[k] = x[k][..., indexs]

  return x


def remove_overlap(result):
  # x = sort_by_scores(result)
  x = copy.deepcopy(result)
  and_mask = copy.deepcopy(x['masks'][:, :, 0])
  for i in range(1, result['masks'].shape[-1]):
    x['masks'][:, :, i] &= (~and_mask)
    and_mask |= x['masks'][:, :, i]

  return x


def rle_encoding(x):
  """ Run-length encoding based on
  https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
  """
  assert x.dtype == np.bool
  dots = np.where(x.flatten() == 1)[0]
  run_lengths = []
  prev = -2
  for b in dots:
    if b > prev + 1:
      run_lengths.append([b, 0])
    run_lengths[-1][1] += 1
    prev = b
  return '|'.join('{} {}'.format(*pair) for pair in run_lengths)


def to_submit_format(image, result, model_type, width=None, height=None):
  final_res = []

  masks     = result['masks']
  class_ids = result['class_ids']
  scores    = result['scores']

  width  = width or masks.shape[1] 
  height = height or masks.shape[0]

  class_names = coco_class if model_type == "coco" else apollo_class

  N = class_ids.shape[0]
  for i in range(N):
    class_id = class_ids[i]
    class_name = class_names[class_id]

    if class_name not in my_class.keys():
      continue
    class_id = my_class[class_name]

    confidence = scores[i]

    cur_mask = masks[:, :, i]
    pixel_count = np.sum(cur_mask)
    if pixel_count == 0:
      continue

    pixel_index = rle_encoding(cur_mask)
  
    final_res.append(("%s,%d,%f,%d,%s" % (image, class_id, confidence, pixel_count, pixel_index)))

  return final_res


def combine_masks(result, model_type, width=None, height=None):
  masks     = result['masks']
  class_ids = result['class_ids']

  width  = width or masks.shape[1]
  height = height or masks.shape[0]

  class_names = coco_class if model_type == "coco" else apollo_class

  mask = np.zeros((height, width), dtype=np.uint16)
  cnt = {
      33 : 0,
      34 : 0,
      35 : 0,
      36 : 0,
      38 : 0,
      39 : 0,
      40 : 0,
    }

  N = class_ids.shape[0]
  for i in range(N):
    class_id = class_ids[i]
    class_name = class_names[class_id]

    if class_name not in my_class.keys():
      continue
    class_id = my_class[class_name]

    mask = np.where(masks[:, :, i], class_id * 1000 + cnt[class_id], mask)
    cnt[class_id] += 1

  return mask


if __name__ == "__main__":
  masks = np.zeros((5, 5, 3), dtype=np.bool)
  masks[:, :, 0] = [
      [False, False, False, False, True ],
      [False, False, False, False, True ],
      [False, False, False, False, True ],
      [False, True,  True,  True,  True ],
      [True,  True,  True,  True,  True ]
    ]
  masks[:, :, 1] = [
      [True,  True,  True,  True,  False],
      [False, True,  True,  False, False],
      [False, True,  True,  False, False],
      [False, True,  True,  False, False],
      [False, True,  True,  False, False]
    ]
  masks[:, :, 2] = [
      [False, False, False, False, False],
      [False, False, False, True,  True ],
      [False, False, False, True,  True ],
      [False, False, False, False, False],
      [False, False, False, False, False]
    ]

  class_ids = np.array([2, 1, 3])
  rois = np.array([2, 1, 3])
  scores = np.array([0.5, 0.8, 0.2])

  result = {
      'rois' : rois,
      'class_ids' : class_ids,
      'scores' : scores,
      'masks' : masks,
    }

  deo = remove_overlap(result)
  
  print(result['scores'],    deo['scores'])
  print(result['rois'],      deo['rois'])
  print(result['class_ids'], deo['class_ids'])
  print(result['masks'][:, :, 0])
  print(result['masks'][:, :, 1])
  print(result['masks'][:, :, 2])
  print("")
  print(deo['masks'][:, :, 0])
  print(deo['masks'][:, :, 1])
  print(deo['masks'][:, :, 2])

