import numpy as np
import os
import copy
from PIL import Image
import open3d as o3d

from readPointCloud import readPointDataAndTransformAngles_mod_colors
from yolo_opencv_images import yolo_segmentation
# from vizualization import Poisson_surface_reconstruction

POINT_NUM = 31
VOXEL_SIZE = 0.05


class PointCollection:
    def __init__(self):
        self.points = None
        self.img = None
        self.info = None
        self.segments = None
        self.transformation_matrix = None


class PointApp:
    def __init__(self):
        self.point_filter = None
        self.img_sermentator = None
        self.point_registrator = None
        self.is_init = False

    def initialize(self, filter, segmentator, registrator) -> bool:
        self.point_filter = filter
        self.img_sermentator = segmentator
        self.point_registrator = registrator

        if self.point_filter is None or self.img_sermentator is None or self.point_registrator is None:
            self.is_init = False
            return False
        else:
            self.is_init = True
            return True

    def point_processing(self, num: int):
        # пока пусть так, через num загрузка
        points_array = self.load_data(num)
        points_array = self.filter(points_array)
        points_array = self.registation(points_array)
        points_array = self.post_process(points_array)

        # на выход подается нужный формат
        pass

    def load_data(self, num: int) -> list:
        # загрузка данных
        print("Загрузка началась")
        points_data = []
        frames_data = []
        for i in range(0, num):
            # получение облаков точек
            model_from_phone, img = readPointDataAndTransformAngles_mod_colors(i)
            frames = Image.fromarray(img).save("yolo/frames/frame_" + str(i) + '.jpg')
            frames_data.append(frames)

            data = PointCollection()
            data.points = model_from_phone
            points_data.append(data)
            # o3d.io.write_point_cloud("pcd/my_room_1/org/model" + str(i) + ".pcd", model_from_phone)
            # print("Оригинальная модель " + str(i) + " сохранена")

        return points_data #надо передать frames_data

    def filter(self, data: list) -> list:
        # фильтрация данных
        if self.point_filter is not None:
            return self.point_filter.filter(data)
        else:
            return data

    def segmentation(self, data: list) -> list:
        # сегментация данных
        if self.img_sermentator is not None:
            return self.img_sermentator.segmentation(data)
        else:
            return data

    def registation(self, data: list) -> list:
        # регистрация данных
        if self.point_registrator is not None:
            return self.point_registrator.registration(data)
        else:
            return data

    def post_process(self, data: list) -> list:
        # постобработка
        return data


class PointFilter:
    def __init__(self):
        pass

    def initialize(self):
        pass

    # фильтрация
    def filter(self, data: list) -> list:
        print("Начало фильтрации")
        out_data = []
        num_it=0
        for cloud in data:
            # глубокое копирование объекта
            temp_data = copy.deepcopy(cloud)
            cl, ind = temp_data.points.remove_statistical_outlier(nb_neighbors=16,
                                                                std_ratio=5.0)

            temp_data.points = cl
            out_data.append(temp_data)

            num_it += 1
            # o3d.io.write_point_cloud("pcd/my_room_1/filter/model" + str(num_it) + ".pcd", cl)
            # print("Модель без шума " + str(num_it) + " сохранена")
        print("Конец фильтрации")
        return out_data

def Poisson_surface_reconstruction(pcd):
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=16,n_threads=-1,scale=1, linear_fit=True)
    print(mesh)
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction()
    print('remove low density vertices')
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(mesh)
    o3d.visualization.draw_geometries([mesh],mesh_show_wireframe = False, mesh_show_back_face=False)
    # Сохранение модели с текстурой и цветом
    # o3d.io.write_triangle_mesh("models/my_room_0.05.glb", mesh, write_vertex_colors =True,write_triangle_uvs =True )

class ImgSegmentator:
    def __init__(self):
        pass

    def initialize(self):
        pass

    # сегментация
    def segmentation(self, data: PointCollection) -> PointCollection:
        # вместо обращения к сохраненным фоткам надо сразу брать переменную frames в load_data()
        # тут пока заготовка

        directory = 'yolo/frames'
        files = os.listdir(directory)

        print("Начало сегментации")
        for img_name in files:
            print(data.img)
            box_coordinates = yolo_segmentation(img_name,directory)
        print("Конец сегментации")


        return data


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                             max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0,
                                             max_nn=100),
    )
    print("Предобработка модели")
    return (pcd_down, pcd_fpfh)

def preprocess_point_cloud_2(pcd, voxel_size):
    pcd_down = pcd
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                             max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0,
                                             max_nn=100),
    )
    print("Доп. обработка модели")
    return (pcd_down, pcd_fpfh)


class PointRegistrator:
    def __init__(self):
        self.max_iteration = 100
        self.confidence = 0.999
        self.voxel_size = 0.05
        self.distance_threshold = 1.5 * self.voxel_size

    def initialize(self):
        pass

    def registration(self, data: list):
        # объединение точек
        print("Начало сшивки")
        if len(data) < 2:
            pass

        # Сложение облаков точек
        # result_cloud = o3d.geometry.PointCloud()
        # clouds = []
        # for i in range(len(data)):
        #     clouds.append(data[i].points)
        # for cloud in clouds:
        #     result_cloud += cloud
        # o3d.visualization.draw(result_cloud)
        #
        # # o3d.io.write_point_cloud("pcd/half_room/half_room1.pcd",result_cloud)
        # # o3d.io.write_point_cloud("pcd/half_room/half_room1.ply", result_cloud)
        # # o3d.io.write_point_cloud("pcd/half_room/half_room1.glt", result_cloud)
        #
        # print("Сохранено")
        # return result_cloud


        dst_down, dst_fpfh = preprocess_point_cloud(data[0].points, self.voxel_size)
        test_source = copy.deepcopy(dst_down)
        transformation_matrix = np.asarray([[0.0, 0.0, 1.0, 0.0],
                                             [1.0, 0.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0, 0.0],
                                             [0.0, 0.0, 0.0, 1.0]])

        for i in range(1, len(data)):
            # src = data[i].points.transform(transformation_matrix)
            src = data[i].points
            # src_down, src_fpfh = preprocess_point_cloud(data[i].points, VOXEL_SIZE)
            src_down, src_fpfh = preprocess_point_cloud(src, VOXEL_SIZE)
            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                src_down,
                dst_down,
                src_fpfh,
                dst_fpfh,
                mutual_filter=True,
                max_correspondence_distance=self.distance_threshold,
                estimation_method=o3d.pipelines.registration.
                    TransformationEstimationPointToPoint(False),
                ransac_n=3,
                checkers=[
                    o3d.pipelines.registration.
                    CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        self.distance_threshold),
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                    self.max_iteration, self.confidence),
            )

            temp_src = copy.deepcopy(src_down)
            test_source = test_source + temp_src
            temp_src.transform(result.transformation)
            # test_source.paint_uniform_color([1, 0.706, 0])
            dst_down = dst_down + temp_src
            dst_down, dst_fpfh = preprocess_point_cloud_2(dst_down, self.voxel_size)
            data[i].transformation_matrix = result.transformation

        # o3d.visualization.draw(dst_down)
        Poisson_surface_reconstruction(dst_down)
        # o3d.io.write_point_cloud("pcd/359/359.pcd",dst_down)
        o3d.io.write_point_cloud("pcd/chear_in_490/chear_in_490.ply", dst_down)
        # print("Сохранено")

        #отсюда должен быть переход данных в сегментацию

        return dst_down


if __name__ == '__main__':
    app = PointApp()

    filter = PointFilter()
    filter.initialize()

    segmentator = ImgSegmentator()
    segmentator.initialize()

    registrator = PointRegistrator()
    registrator.initialize()


    app.initialize(filter, segmentator, registrator)

    app.point_processing(POINT_NUM)

