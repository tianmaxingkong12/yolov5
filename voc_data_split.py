import xml.etree.ElementTree as ET

from tqdm import tqdm
from utils.general import download, Path

voc_names = {
  0: "aeroplane",
  1: "bicycle",
  2: "bird",
  3: "boat",
  4: "bottle",
  5: "bus",
  6: "car",
  7: "cat",
  8: "chair",
  9: "cow",
  10: "diningtable",
  11: "dog",
  12: "horse",
  13: "motorbike",
  14: "person",
  15: "pottedplant",
  16: "sheep",
  17: "sofa",
  18: "train",
  19: "tvmonitor"
}


def convert_label(path, lb_path, year, image_id):
    def convert_box(size, box):
          dw, dh = 1. / size[0], 1. / size[1]
          x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
          return x * dw, y * dh, w * dw, h * dh

    in_file = open(path / f'Annotations/{image_id}.xml')
    out_file = open(lb_path, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    names = list(voc_names.values())  # names list
    for obj in root.iter('object'):
          cls = obj.find('name').text
          if cls in names and int(obj.find('difficult').text) != 1:
              xmlbox = obj.find('bndbox')
              bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
              cls_id = names.index(cls)  # class id
              out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')


# Download
dir = Path("./datasets/VOC")  # dataset root dir
# url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/'
# urls = [f'{url}VOCtrainval_06-Nov-2007.zip',  # 446MB, 5012 images
#         f'{url}VOCtest_06-Nov-2007.zip',  # 438MB, 4953 images
#         f'{url}VOCtrainval_11-May-2012.zip']  # 1.95GB, 17126 images
# download(urls, dir=dir / 'images', delete=False, curl=True, threads=3)

# Convert
# path = dir / 'images/VOCdevkit'
source_path = Path("./datasets/VOC07_12/")
for year, image_set in ('0712', 'train'),('0712','val'):
      imgs_path = dir / 'images' / f'{image_set}{year}'
      lbs_path = dir / 'labels' / f'{image_set}{year}'
      imgs_path.mkdir(exist_ok=True, parents=True)
      lbs_path.mkdir(exist_ok=True, parents=True)
      
      with open(source_path / f'ImageSets/Main/{image_set}{year}.txt') as f:
          image_ids = f.read().strip().split()
      for id in tqdm(image_ids, desc=f'{image_set}{year}'):
          f = source_path / f'JPEGImages/{id}.jpg'  # old img path
          lb_path = (lbs_path / f.name).with_suffix('.txt')  # new label path
          f.rename(imgs_path / f.name)  # move image
          convert_label(source_path, lb_path, year, id)  # convert labels to YOLO format