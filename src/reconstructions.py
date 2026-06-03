from pathlib import Path
from deepdiff import DeepDiff
from src.utils import new_numbered_folder
from src.config import get_all_configs
import yaml
import logging
import pycolmap
import h5py
logger = logging.getLogger("CleftReconstruction")
import os
import shutil
import random
import torch
import tempfile
from contextlib import nullcontext
import functools
from src.demo_glomap_custom import get_args_parser, main_demo, get_reconstructed_scene
from thirdparty.Mast3r.mast3r.model import AsymmetricMASt3R
from thirdparty.Mast3r.mast3r.utils.misc import hash_md5

import mast3r.utils.path_to_dust3r  # noqa
from thirdparty.Mast3r.dust3r.demo import set_print_with_timestamp
import matplotlib.pyplot as pl
pl.ion()
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12



def mast3r_reconstruction(subsample_path, rec_dir,config):
    parser = get_args_parser()
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.device = 'cpu'
    set_print_with_timestamp()

    random.seed(config['random_seed'])
    file_list = [str(os.path.abspath(i)) for i in list(Path(subsample_path).glob("*.png"))]
    args.model_name = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    args.retrieval_model = "./thirdparty/Mast3r/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
    args.weights = None
    options = pycolmap.IncrementalPipelineOptions()
    tri_ops = pycolmap.IncrementalTriangulatorOptions()
    # tri_ops.complete_max_reproj_error = 2.0
    # tri_ops.merge_max_reproj_error = 2.0
    confidence = 1.001
    options.triangulation = tri_ops
    dense_matching = False
    num_matched =15
    skip_geometric_verification = False



    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

    if args.weights is not None:
        weights_path = args.weights
    else:
        weights_path = "naver/" + args.model_name

    model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)
    chkpt_tag = hash_md5(weights_path)

    def get_context(tmp_dir):
        return tempfile.TemporaryDirectory(suffix='_mast3r_gradio_demo') if tmp_dir is None \
            else nullcontext(tmp_dir)

    cache_path = rec_dir
    os.makedirs(cache_path, exist_ok=True)
    recon_fun = functools.partial(get_reconstructed_scene, args.glomap_bin, cache_path, args.gradio_delete_cache, model,
                                  args.retrieval_model, args.device, args.silent, args.image_size)
    recon_fun(current_scene_state=None, filelist=file_list, transparent_cams=False, cam_size=0.01,
                                scenegraph_type=config['scenegraph_type'], winsize=3, win_cyclic=False, refid=0,
                                shared_intrinsics=False, dense_matching=dense_matching, conf_thr=confidence,
                                skip_geometric_verification=skip_geometric_verification,
                                num_matched=15, options=options)
    rec = pycolmap.Reconstruction(str(Path(rec_dir)/ 'reconstruction/0'))
    rec.export_PLY(str(Path(rec_dir) / 'sparse.ply'))