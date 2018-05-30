import cv2

# img = cv2.imread("/nfs/project/CVPR_WAD2018/ch4_instance_segmentation/test/053c8b05f1afb2aa983d9da762ce7b10.jpg")
img_file = None

import fileinput
cnt = 0
for line in fileinput.input("res_20180508.csv"):
  if cnt == 0:
    cnt += 1
    continue

  cur = line.strip().split(',')[0]
  if not img_file:
    img_file = cur
    img = cv2.imread("/nfs/project/CVPR_WAD2018/ch4_instance_segmentation/test/" + img_file)
  elif img_file != cur:
    img = cv2.resize(img, (1024, 1024))
    cv2.imshow("1", img)
    cv2.waitKey(0)
    img_file = cur
    img = cv2.imread("/nfs/project/CVPR_WAD2018/ch4_instance_segmentation/test/" + img_file)    
  
  indexs = line.strip().split(',')[-1].split("|")
  for x in indexs[:-2]:
    s, l = map(int, x.split(" "))
    for i in range(l):
      r = int((s + i) / 3384)
      c = int((s + i) % 3384)
      cv2.circle(img, (c, r), 1, (0, 255, 0), -1)

