import numpy as np
import os
import cv2
import time
import sys
from ultralytics import YOLO
import open3d as o3d
import pandas as pd


HEIGHT = 160
WIDTH = 90

CONFIDENCE = 0.01
font_scale = 1
thickness = 1

model = YOLO("yolo/models/best.pt")
labels = open("yolo/data/coco.names").read().strip().split("\n")
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

def yolo_segmentation(img_name,directory):
    path_name, _ = os.path.splitext(directory)
    directory += "-segmentation"
    if not os.path.isdir(directory):
        os.mkdir(directory)
    image = cv2.imread(path_name + "/" + img_name)
    name = os.path.basename(img_name)
    filename, ext = name.split(".")
    start = time.perf_counter()
    results = model.predict(image,show=False,conf=CONFIDENCE)[0]
    time_took = time.perf_counter() - start
    print(f"Time took: {time_took:.2f}s")

    box_coordinates = []
    for data in results.boxes.data.tolist():
        xmin, ymin, xmax, ymax, confidence, class_id = data
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        class_id = int(class_id)

        box_coordinates.append([xmin, ymin, xmax, ymax])

        color = [int(c) for c in colors[class_id]]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=color, thickness=thickness)
        text = f"{labels[class_id]}: {confidence:.2f}"

        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
        text_offset_x = xmin
        text_offset_y = ymin - 5
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
        overlay = image.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)

        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

        cv2.putText(image, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

    saveimage_name = os.path.join(directory ,filename + "_yolo8." + ext)
    # cv2.imwrite(saveimage_name, image)
    return box_coordinates

directory = 'yolo/frames_1'
files = os.listdir(directory)

get_cor = []
for img_name in files:
    info = yolo_segmentation(img_name,directory)
    get_cor.append(info)

indices = []
for obj in get_cor:
    for val in obj:
        xmin, ymin, xmax, ymax = val
        # Расчет индексов
        for i in range(ymin,ymax):
            indices.append(np.arange(i*WIDTH + xmin,i*WIDTH + xmax))

print("Индексы: ","\n", pd.DataFrame(indices))
# print(print("\n","Общий вид: ",'\n'.join(['\t'.join([str(cell) for cell in row]) for row in indices])))

pcd = o3d.io.read_point_cloud('pcd/my_room/my_room_0.05.ply')
# point_list = [pcd]
point_list = []
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    # outlier_cloud = cloud.select_by_index(ind, invert=True) # Set to True to save points other than ind
    print("Showing outliers (red) and inliers : ")
    # outlier_cloud.paint_uniform_color([0, 0, 0])
    inlier_cloud.paint_uniform_color([1, 0, 0])
    point_list.append(inlier_cloud)
    # point_list.append(outlier_cloud)

print(len(indices))


for i in indices[:3000]:
    # print(type(i),"\n", i.flatten())
    display_inlier_outlier(pcd, i)


o3d.visualization.draw(point_list)


