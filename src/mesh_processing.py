import logging
import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial import cKDTree

logger = logging.getLogger("CleftReconstruction")


def color_filter(pcd, min_val=1.0, max_val=253.0):
    colors_255 = np.asarray(pcd.colors) * 255
    composite = np.mean(colors_255, axis=1)
    mask = (composite >= min_val) & (composite <= max_val)
    return pcd.select_by_index(np.where(mask)[0])


def noise_filter(pcd, nb_neighbors=10, std_ratio=1.5):
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(ind)


def sor_filter(pcd, nb_neighbors=20, std_ratio=2.0, n_passes=3):
    for _ in range(n_passes):
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        pcd = pcd.select_by_index(ind)
    return pcd


def mls_smooth(pcd, search_radius=0.7):
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    tree = cKDTree(pts)
    pairs = tree.query_ball_tree(tree, r=search_radius)
    new_pts = np.array([pts[idx].mean(axis=0) for idx in pairs])
    new_cols = np.array([cols[idx].mean(axis=0) for idx in pairs])
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(new_pts)
    out.colors = o3d.utility.Vector3dVector(np.clip(new_cols, 0.0, 1.0))
    return out


def estimate_normals(pcd):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    # Orient normals away from a far -Z camera, matching CloudCompare MINUS_Z preference
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, -1000.0]))
    return pcd


def interpolate_colors(source_pcd, target_pcd):
    source_tree = o3d.geometry.KDTreeFlann(source_pcd)
    source_colors = np.asarray(source_pcd.colors)
    target_points = np.asarray(target_pcd.points)
    new_colors = np.zeros((len(target_points), 3))
    for i, p in enumerate(target_points):
        _, idx, _ = source_tree.search_knn_vector_3d(p, 1)
        new_colors[i] = source_colors[idx[0]]
    target_pcd.colors = o3d.utility.Vector3dVector(new_colors)
    return target_pcd


def poisson_reconstruct(pcd, depth=9, density_quantile=0.1):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=1.1, linear_fit=False
    )
    vertices_to_remove = np.asarray(densities) < np.quantile(np.asarray(densities), density_quantile)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    return mesh


def denoise_point_cloud(ply_path, output_dir, conf):
    """
    Runs the configured denoising steps on the point cloud.
    Returns the path to the denoised PLY (fed to NKSR).
    If all DO_* flags are False, returns ply_path unchanged.

    Data flow:
        ply_path → estimate_normals → saved as og_<name>.ply  [reference]
                 → [color_filter]
                 → [noise_filter]
                 → [sor_filter]
                 → saved as denoised_<name>.ply  ← returned; fed to NKSR
    """
    output_dir = Path(output_dir)
    name = Path(ply_path).parent.name

    any_filter = conf['do_color_filter'] or conf['do_noise_filter'] or conf['do_sor']
    if not any_filter:
        logger.info("Denoising: all filters disabled, skipping")
        return str(ply_path)

    pcd = o3d.io.read_point_cloud(str(ply_path))
    logger.info(f"Denoising: loaded {len(pcd.points)} points from {ply_path}")

    pcd = estimate_normals(pcd)
    o3d.io.write_point_cloud(str(output_dir / f"og_{name}.ply"), pcd)

    if conf['do_color_filter']:
        pcd = color_filter(pcd, min_val=conf['color_filter_min'], max_val=conf['color_filter_max'])
        logger.info(f"Denoising: after color filter: {len(pcd.points)} points")

    if conf['do_noise_filter']:
        pcd = noise_filter(pcd, nb_neighbors=conf['noise_nb_neighbors'], std_ratio=conf['noise_std_ratio'])
        logger.info(f"Denoising: after noise filter: {len(pcd.points)} points")

    if conf['do_sor']:
        pcd = sor_filter(pcd, nb_neighbors=conf['sor_nb_neighbors'], std_ratio=conf['sor_std_ratio'],
                         n_passes=conf['sor_passes'])
        logger.info(f"Denoising: after SOR: {len(pcd.points)} points")

    out_path = output_dir / f"denoised_{name}.ply"
    o3d.io.write_point_cloud(str(out_path), pcd)
    logger.info(f"Denoising: saved denoised cloud to {out_path}")
    return str(out_path)


def mls_and_poisson(denoised_ply_path, output_dir, mls_conf, poisson_conf):
    """
    Runs MLS smoothing and/or Poisson reconstruction on the denoised cloud.

    Data flow:
        denoised_ply_path → [mls_smooth] → smoothed_<name>.ply
                         → [poisson_reconstruct] → poisson_mesh_<name>.ply
    """
    if not mls_conf['do_mls'] and not poisson_conf['do_poisson']:
        return

    output_dir = Path(output_dir)
    name = Path(denoised_ply_path).parent.name

    pcd = o3d.io.read_point_cloud(str(denoised_ply_path))
    denoised_pcd = pcd

    if mls_conf['do_mls']:
        pcd = mls_smooth(pcd, search_radius=mls_conf['search_radius'])
        pcd = estimate_normals(pcd)
        pcd = interpolate_colors(denoised_pcd, pcd)
        smoothed_path = output_dir / f"smoothed_{name}.ply"
        o3d.io.write_point_cloud(str(smoothed_path), pcd)
        logger.info(f"MLS: saved smoothed cloud to {smoothed_path} ({len(pcd.points)} points)")

    if poisson_conf['do_poisson']:
        if not pcd.has_normals():
            pcd = estimate_normals(pcd)
        mesh = poisson_reconstruct(pcd, depth=poisson_conf['depth'],
                                   density_quantile=poisson_conf['density_quantile'])
        mesh_path = output_dir / f"poisson_mesh_{name}.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        logger.info(f"Poisson: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles → {mesh_path}")
