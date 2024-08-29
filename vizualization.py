import open3d as o3d
import numpy as np
import trimesh
from scipy.spatial import Delaunay

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
    o3d.visualization.draw_geometries([mesh],mesh_show_wireframe = False, mesh_show_back_face=True)
    #Сохранение модели с текстурой и цветом
    # o3d.io.write_triangle_mesh("models/my_room_0.05.glb", mesh, write_vertex_colors =True,write_triangle_uvs =True )


def Ball_pivoting(pcd):
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction()

    # estimate radius for rolling ball
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 2 * avg_dist
    print("avg_dist = ",avg_dist," radius = ",radius)
    # avg_dist =  0.023536557029252798  radius =  0.09414622811701119
    print("radius*2 = ",radius * 2," radius*3 = ", radius * 3)
    # radius*2 =  0.18829245623402238  radius*3 =  0.2824386843510336

    radii = [0.05, 0.02, 0.03, 0.04]
    # radii = [radius, radius * 2, radius * 3]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(radii))

    # pcd1 = mesh.sample_points_poisson_disk(14400)
    # pcd1.paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([pcd1])
    # create the triangular mesh with the vertices and faces from open3d
    tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                               vertex_normals=np.asarray(mesh.vertex_normals))

    trimesh.convex.is_convex(tri_mesh)
    o3d.visualization.draw_geometries([mesh],mesh_show_wireframe = False, mesh_show_back_face=True)



def Alpha_forms(pcd):
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    alpha = 0.05
    print(f"alpha={alpha:.3f}")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha, tetra_mesh, pt_map)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_wireframe =True, mesh_show_back_face=True)

    # tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    # for alpha in np.logspace(np.log10(0.05), np.log10(0.03), num=4):
    #     print(f"alpha={alpha:.3f}")
    #     mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
    #         pcd, alpha, tetra_mesh, pt_map)
    #     mesh.compute_vertex_normals()
    #     o3d.visualization.draw_geometries([mesh],mesh_show_wireframe =True, mesh_show_back_face=True)


pcd = o3d.io.read_point_cloud('pcd/my_room/my_room_0.05.ply')
Poisson_surface_reconstruction(pcd)
# Ball_pivoting(pcd)
# Alpha_forms(pcd)