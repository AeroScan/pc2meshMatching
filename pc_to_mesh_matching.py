import numpy as np
import open3d as o3d
from textual2pcd import write_pcd_by_pc

def writeTXT(filename, points, normals, labels):
    pcd = np.zeros((points.shape[0], 8))
    pcd[:,0:3] = points
    pcd[:, 3:6] = normals
    pcd[:, 7] = labels

    np.savetxt(filename, pcd, fmt='%f') 

scene = o3d.t.geometry.RaycastingScene()

mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh("70_10.obj"))

scene.add_triangles(mesh)

pcd = o3d.io.read_point_cloud("70_10.pcd")

ans = scene.compute_closest_points(o3d.core.Tensor(np.asarray(pcd.points), dtype=o3d.core.Dtype.Float32))

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

write_pcd_by_pc(pc, '70_10_match.pcd', ['x','y','z','normal_x','normal_y','normal_z','label'], binary=False)