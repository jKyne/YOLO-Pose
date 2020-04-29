import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join



def Calc_Centroid(polygon):
  totalArea = 0
  totalX = 0
  totalY = 0

  point_num = 0
  while (point_num < 5 and not(polygon[point_num][0] == 0 and polygon[point_num][1] == 0)):
      point_num += 1


  for i in range(point_num):
    a = polygon[(i + 1)%point_num]
    b = polygon[i]

    area = 0.5 * (a[0] * b[1] - b[0] * a[1]) 
    x = (a[0] + b[0]) / 3 
    y = (a[1] + b[1]) / 3 

    totalArea += area
    totalX += area * x
    totalY += area * y
  
  return [totalX / totalArea, totalY/ totalArea]


  
def convert(size, polygon):
  dw = 1./(size[0])
  dh = 1./(size[1])
  x1 = polygon[0][0]*dw
  y1 = polygon[0][1]*dh
  x2 = polygon[1][0]*dw
  y2 = polygon[1][1]*dh
  x3 = polygon[2][0]*dw
  y3 = polygon[2][1]*dh
  x4 = polygon[3][0]*dw
  y4 = polygon[3][1]*dh
  x5 = polygon[4][0]*dw
  y5 = polygon[4][1]*dh
  (centr_x , centr_y) = Calc_Centroid(polygon)
  x0 = centr_x*dw
  y0 = centr_y*dh

  return (x0,y0,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5)

def convert_annotation(image_id):
    in_file = open('data/Annotations/%s.xml'%image_id)
    out_file = open('data/labels/%s.txt'%image_id, 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        polygon = obj.find('polygon')
        p = [([0] * 2) for i in range(5)]
        for i in range(1 , 6):
            point = [polygon.find('x%s'%str(i)) , polygon.find('y%s'%str(i))]
            if point[0] != None:
                point[0] = float(point[0].text)
                point[1] = float(point[1].text)
                p[i-1] = point
        pp = convert((w,h), p)

        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        xmax = float(bndbox.find('xmax').text)
        ymin = float(bndbox.find('ymin').text)
        ymax = float(bndbox.find('ymax').text)
        x_range = str((xmax - xmin)/w)
        y_range = str((ymax - ymin)/h)


        out_file.write("0 " + " ".join([str(a) for a in pp]) + " " + x_range + " " + y_range + '\n')




if __name__ == '__main__':
    wd = getcwd()
    classes = ['ignore']
    total_poseLabel = os.listdir("data/labels_pose")
    for i in range(len(total_poseLabel)):
        total_poseLabel[i] = total_poseLabel[i].strip('.txt')

    for image_set in ['train','test']:
      if not os.path.exists('data/labels/'):
          os.makedirs('data/labels/')
      image_ids = open('data/ImageSets/Main/%s.txt'%image_set).read().strip().split()
      list_file = open('data/%s.txt'%image_set, 'w')
      for image_id in image_ids:
          if image_id in total_poseLabel:
              list_file.write('%s/data/JPEGImages/%s.jpeg\n'%(wd, image_id))
              convert_annotation(image_id)
      list_file.close()

      
