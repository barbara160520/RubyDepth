import numpy as np
# import pclpy
# from pclpy import pcl
import struct
import open3d as o3d
import keyboard


def getFloatFromBit(sym):
    # Assuming sym is a byte array or bytes object
    if len(sym) != 4:  # Check if it has 4 bytes (32 bits)
        raise ValueError("Input must be 4 bytes long")

    # Unpack the bytes as a single precision (32-bit) float
    f = struct.unpack('@f', sym)[0]
    return f

def getIntFromBit(sym):
    # Assuming sym is a byte array or bytes object
    if len(sym) != 4:  # Check if it has 4 bytes (32 bits)
        raise ValueError("Input must be 4 bytes long")

    # Unpack the bytes as a single precision (32-bit) float
    f = struct.unpack('@i', sym)[0]
    return f


# def readPointDataAndTransformAngles(num):
#     pd_filename = f"./point_{num}.txt"
#     t_filename = f"./t_{num}.txt"
#     r_filename = f"./r_{num}.txt"
#
#     transform_file = open(t_filename, "rb")
#     point_file = open(pd_filename, "rb")
#     rotate_file = open(r_filename, "rb")
#
#     raw_point_data = []
#
#     sym = transform_file.read(4)
#     x_t = getFloatFromBit(sym)
#     sym = transform_file.read(4)
#     y_t = getFloatFromBit(sym)
#     sym = transform_file.read(4)
#     z_t = getFloatFromBit(sym)
#
#     sym = rotate_file.read(4)
#     a1 = getFloatFromBit(sym)
#     sym = rotate_file.read(4)
#     b1 = getFloatFromBit(sym)
#     sym = rotate_file.read(4)
#     g1 = getFloatFromBit(sym)
#
#     a_alpha = -b1
#     a_beta = -a1
#     a_gamma = -g1
#
#     transform_matrix = np.identity(4, dtype=np.float32)
#     transform_matrix[0, 0] = np.cos(a_beta) * np.cos(a_gamma)
#     transform_matrix[0, 1] = -np.sin(a_gamma) * np.cos(a_beta)
#     transform_matrix[0, 2] = np.sin(a_beta)
#
#     transform_matrix[1, 0] = (
#         np.sin(a_alpha) * np.sin(a_beta) * np.cos(a_gamma) + np.sin(a_gamma) * np.cos(a_alpha)
#     )
#     transform_matrix[1, 1] = (
#         -np.sin(a_alpha) * np.sin(a_beta) * np.sin(a_gamma)
#         + np.cos(a_alpha) * np.cos(a_gamma)
#     )
#     transform_matrix[1, 2] = -np.sin(a_alpha) * np.cos(a_beta)
#
#     transform_matrix[2, 0] = (
#         np.sin(a_alpha) * np.sin(a_gamma)
#         - np.sin(a_beta) * np.cos(a_alpha) * np.cos(a_gamma)
#     )
#     transform_matrix[2, 1] = (
#         np.sin(a_alpha) * np.cos(a_gamma)
#         + np.sin(a_beta) * np.sin(a_gamma) * np.cos(a_alpha)
#     )
#     transform_matrix[2, 2] = np.cos(a_alpha) * np.cos(a_beta)
#
#     transform_matrix[0, 3] = y_t
#     transform_matrix[1, 3] = x_t
#     transform_matrix[2, 3] = z_t
#
#     while True:
#         sym = point_file.read(4)
#         if not sym:
#             break
#         val = getFloatFromBit(sym)
#         raw_point_data.append(val)
#
#     raw_point_data_len = len(raw_point_data) // 4
#
#     points = pclpy.pcl.PointCloud.PointXYZ()
#     # cloud = pclpy.pcl.PointCloud.PointXYZ()
#     # points.append(cloud(x, y, z))
#     # points = []
#     count = 0
#     temp_point = pclpy.pcl.point_types.PointXYZ()
#     for i in range(raw_point_data_len):
#         x = raw_point_data[count] + 0
#         y = raw_point_data[count + 1] + 0
#         z = raw_point_data[count + 2] + 0
#         rgb = raw_point_data[count + 3]
#
#         if (
#             np.isinf(x)
#             or np.isinf(y)
#             or np.isinf(z)
#             or np.isnan(x)
#             or np.isnan(y)
#             or np.isnan(z)
#             # or x > 2
#             # or y > 2
#             # or z > 2
#             # or x < -2
#             # or y < -2
#             # or z < -2
#         ):
#             continue
#
#         temp_point.x = x
#         temp_point.y = y
#         temp_point.z = z
#
#         points.push_back(temp_point)
#         count += 4
#
#     pcl.common.transformPointCloud(points, points, transform_matrix)
#     return points

def readPointDataAndTransformAngles_mod_colors(num):
    pd_filename = f"./points/359/point_{num}.txt"
    t_filename = f"./points/359/t_{num}.txt"
    r_filename = f"./points/359/r_{num}.txt"
    c_filename = f"./points/359/colors_{num}.txt"

    transform_file = open(t_filename, "rb")
    point_file = open(pd_filename, "rb")
    rotate_file = open(r_filename, "rb")
    colors_file = open(c_filename, "rb")

    raw_point_data = []
    colors_data = []

    sym = transform_file.read(4)
    x_t = getFloatFromBit(sym)
    sym = transform_file.read(4)
    y_t = getFloatFromBit(sym)
    sym = transform_file.read(4)
    z_t = getFloatFromBit(sym)

    sym = rotate_file.read(4)
    a1 = getFloatFromBit(sym)
    sym = rotate_file.read(4)
    b1 = getFloatFromBit(sym)
    sym = rotate_file.read(4)
    g1 = getFloatFromBit(sym)

    a_alpha = -b1
    a_beta = -a1
    a_gamma = -g1

    transform_matrix = np.identity(4, dtype=np.float32)
    transform_matrix[0, 0] = np.cos(a_beta) * np.cos(a_gamma)
    transform_matrix[0, 1] = -np.sin(a_gamma) * np.cos(a_beta)
    transform_matrix[0, 2] = np.sin(a_beta)

    transform_matrix[1, 0] = (
        np.sin(a_alpha) * np.sin(a_beta) * np.cos(a_gamma) + np.sin(a_gamma) * np.cos(a_alpha)
    )
    transform_matrix[1, 1] = (
        -np.sin(a_alpha) * np.sin(a_beta) * np.sin(a_gamma)
        + np.cos(a_alpha) * np.cos(a_gamma)
    )
    transform_matrix[1, 2] = -np.sin(a_alpha) * np.cos(a_beta)

    transform_matrix[2, 0] = (
        np.sin(a_alpha) * np.sin(a_gamma)
        - np.sin(a_beta) * np.cos(a_alpha) * np.cos(a_gamma)
    )
    transform_matrix[2, 1] = (
        np.sin(a_alpha) * np.cos(a_gamma)
        + np.sin(a_beta) * np.sin(a_gamma) * np.cos(a_alpha)
    )
    transform_matrix[2, 2] = np.cos(a_alpha) * np.cos(a_beta)

    transform_matrix[0, 3] = y_t
    transform_matrix[1, 3] = x_t
    transform_matrix[2, 3] = z_t
    transform_matrix_o3d = o3d.core.Tensor(transform_matrix)

    while True:
        sym = point_file.read(4)
        if not sym:
            break
        val = getFloatFromBit(sym)
        raw_point_data.append(val)

    # получение данных о размере изображения
    depth_w = 0
    depth_h = 0
    sym = colors_file.read(4)
    if sym:
        val = getIntFromBit(sym)
        depth_w = val

    sym = colors_file.read(4)
    if sym:
        val = getIntFromBit(sym)
        depth_h = val

    while True:
        sym = colors_file.read(4)
        if not sym:
            break
        val = getFloatFromBit(sym)
        colors_data.append(val)

    color_img = np.zeros([depth_w, depth_h, 3], dtype=np.uint8)


    raw_point_data_len = len(raw_point_data) // 4

    # points = pclpy.pcl.PointCloud.PointXYZ()
    points = o3d.geometry.PointCloud()
    # cloud = pclpy.pcl.PointCloud.PointXYZ()
    # points.append(cloud(x, y, z))
    # points = []
    count = 0
    colors_cnt = 0
    col_cnt = 0
    row_cnt = 0
    zero_point_cnt = 0
    # temp_point = pclpy.pcl.point_types.PointXYZ()
    for i in range(raw_point_data_len):
        x = raw_point_data[count] + 0
        y = raw_point_data[count + 1] + 0
        z = raw_point_data[count + 2] + 0
        rgb = raw_point_data[count + 3]

        # if np.abs(z) > 0.001:
        #     r = colors_data[colors_cnt] + 0
        #     g = colors_data[colors_cnt + 1] + 0
        #     b = colors_data[colors_cnt + 2] + 0
        #     colors_cnt += 3
        # else:
        #     r = 0
        #     g = 0
        #     b = 0
        #     zero_point_cnt += 1

        r = colors_data[colors_cnt] + 0
        g = colors_data[colors_cnt + 1] + 0
        b = colors_data[colors_cnt + 2] + 0
        colors_cnt += 3

        temp_color = np.array([[r, g, b]])
        color_img[col_cnt, row_cnt, 0] = np.uint8(r * 255)
        color_img[col_cnt, row_cnt, 1] = np.uint8(g * 255)
        color_img[col_cnt, row_cnt, 2] = np.uint8(b * 255)
        col_cnt += 1
        if col_cnt >= depth_w:
            col_cnt = 0
            row_cnt += 1

        if (
            np.isinf(x)
            or np.isinf(y)
            or np.isinf(z)
            or np.isnan(x)
            or np.isnan(y)
            or np.isnan(z)
            or z == 0
            # or x > 2
            # or y > 2
            # or z > 2
            # or x < -2
            # or y < -2
            # or z < -2
        ):
            temp_point = np.array([[0, 0, 0]])
            points.points.extend(temp_point)
            points.colors.extend(temp_color)
            count += 4
            continue

        temp_point = np.array([[x, y, z]])

        # temp_point.x = x
        # temp_point.y = y
        # temp_point.z = z

        # points.push_back(temp_point)
        points.points.extend(temp_point)
        points.colors.extend(temp_color)
        count += 4

        # print(i,zero_point_cnt)
    transform_file.close()
    point_file.close()
    rotate_file.close()
    colors_file.close()


    points.transform(transform_matrix)

    return points, color_img

