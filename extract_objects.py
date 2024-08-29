import numpy as np
import os
from yolo_opencv_images import yolo_segmentation
import pandas as pd
import open3d as o3d

h = 160
w = 90

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
            indices.append(np.arange(i*w + xmin,i*w + xmax))

# print("Индексы: ","\n", pd.DataFrame(indices))
# print(print("\n","Общий вид: ",'\n'.join(['\t'.join([str(cell) for cell in row]) for row in indices])))

pcd = o3d.io.read_point_cloud('pcd/my_room/my_room_0.05.ply')
point_list = []
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    # outlier_cloud = cloud.select_by_index(ind, invert=True)
    # Set to True to save points other than ind
    print("Showing outliers (red) and inliers : ")
    # outlier_cloud.paint_uniform_color([0, 1, 0])
    inlier_cloud.paint_uniform_color([1, 0, 0])
    point_list.append(inlier_cloud)
    # point_list.append(outlier_cloud)


for i in indices:
    # print(type(i),"\n", i.flatten())
    display_inlier_outlier(pcd, i)

print(point_list)
o3d.visualization.draw(point_list)





