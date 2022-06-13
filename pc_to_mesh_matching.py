import numpy as np
import open3d as o3d
from textual2pcd import write_pcd_by_pc

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform a matching of a point cloud and a mesh, saving as PCD using the field \'lable\' to identify the triangle id for each point.')
    parser.add_argument('input_filename_pc', type=str, help='input filename of point cloud in .pcd.')
    parser.add_argument('input_filename_mesh', type=str, help='input filename of mesh in .obj.')
    parser.add_argument('output_filename', type=str, help='output filename in .pcd.')
    parser.add_argument('-t', '--threads', type=int, default=0, help='number of threads to be used.')
    parser.add_argument('-b', '--binary', action='store_true', help='save in binary format. Default = False')
    args = vars(parser.parse_args())

    input_filename_pc = args['input_filename_pc']
    input_filename_mesh = args['input_filename_mesh']
    output_filename = args['output_filename']
    threads = args['threads']
    binary = args['binary']

    scene = o3d.t.geometry.RaycastingScene()
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(input_filename_mesh))
    scene.add_triangles(mesh)

    pcd = o3d.io.read_point_cloud(input_filename_pc)

    ans = scene.compute_closest_points(o3d.core.Tensor(np.asarray(pcd.points), dtype=o3d.core.Dtype.Float32), nthreads=threads)

    points = np.asarray(pcd.points)
    aprox_points = ans['points'].numpy()
    mean_dist = np.mean(np.sqrt(np.power(points - aprox_points, 2)))

    invalid = 0
    for elem in ans['primitive_ids'].numpy():
        if elem == o3d.t.geometry.RaycastingScene.INVALID_ID:
            invalid += 1

    print('Mesh to Points mean distance:', mean_dist)
    print('No collision points:', invalid)

    pc = np.zeros((points.shape[0], 7))
    pc[:,0:3] = points
    pc[:, 3:6] = np.asarray(pcd.normals)
    pc[:, 6] = ans['primitive_ids'].numpy()

    write_pcd_by_pc(pc, output_filename, ['x','y','z','normal_x','normal_y','normal_z','label'], binary=binary)