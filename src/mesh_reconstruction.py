import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "thirdparty" / "nksr"))

import logging
import nksr
import torch
import open3d as o3d
import numpy as np
import trimesh

logger = logging.getLogger("CleftReconstruction")


def nksr_reconstruction(ply_path, output_dir, nksr_conf):
    logger.info(f"Starting NKSR mesh reconstruction from {ply_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pcd = o3d.io.read_point_cloud(str(ply_path))
    xyz = torch.from_numpy(np.asarray(pcd.points)).float().to(device)

    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
    normals = torch.from_numpy(np.asarray(pcd.normals)).float().to(device)

    reconstructor = nksr.Reconstructor(device)
    field = reconstructor.reconstruct(
        xyz,
        normal=normals,
        detail_level=nksr_conf['detail_level'],
    )
    mesh = field.extract_dual_mesh(mise_iter=nksr_conf['mise_iter'])

    out_path = Path(output_dir) / nksr_conf['output_name']
    output_mesh = trimesh.Trimesh(
        vertices=mesh.v.cpu().numpy(),
        faces=mesh.f.cpu().numpy(),
    )
    output_mesh.export(str(out_path))
    logger.info(f"Mesh saved to {out_path}")
    return str(out_path)
