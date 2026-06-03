from yacs.config import CfgNode as CN
import cv2
import yaml
import glob
from pathlib import Path
import pprint

def print_conf(conf):
    return pprint.pformat(dict(conf))

def get_all_configs(path, get_file_path=False, config_name="cur_config.yaml"):
    configs = []
    for filename in glob.iglob(path + '/**/' + config_name, recursive=True):
        with open(filename, 'r') as file:
            test = yaml.load(file, Loader=yaml.Loader)
            parent_path = Path(filename).parent
            if get_file_path:
                parent_path = Path(filename)
            configs.append((parent_path, dict(test)))
    return configs

def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}

def generate_config(vid, name, experiment_config):

    _CN = CN()
    _CN.VERSION = 1.0
    _CN.PATHS = CN()
    _CN.VIDEO_CONVERSION = CN()
    _CN.FRAME_SCALING = CN()
    _CN.SAMPLING = CN()
    _CN.HIGHLIGHT_REMOVAL = CN()
    _CN.MAST3R = CN()
    _CN.POSTPROCESSING = CN()


    _CN.PATHS.INPUT_VIDEO_DIR = 'your path here'
    _CN.PATHS.INPUT_VIDEO_PATH = vid
    _CN.PATHS.EXPERIMENT_PATH = '../output/experiments'
    _CN.PATHS.OUTPUT_ROOT_PATH = _CN.PATHS.EXPERIMENT_PATH + "/" +f'{name}'
    #Folder of already existing masks
    _CN.PATHS.MASK_MIRROR_PATH = '../output/video_masks/' +f'{name}' +"/mask"

    _CN.PATHS.ROOT_FRAME_PATH = _CN.PATHS.OUTPUT_ROOT_PATH + "/" +'frames'
    _CN.VIDEO_CONVERSION.MIRROR_IMAGES = False

    _CN.PATHS.MASKS_PATH = _CN.PATHS.OUTPUT_ROOT_PATH + "/" +"masks"
    _CN.PATHS.HIGHLIGHT_MASKS_ROOT = _CN.PATHS.OUTPUT_ROOT_PATH+ "/" +"masks/highlight_masks"
    _CN.HIGHLIGHT_REMOVAL.THRESHOLD = 255

    _CN.PATHS.COMBINED_MASKS = _CN.PATHS.MASKS_PATH + "/combined_masks"
    _CN.PATHS.COMBINED_MASKS_IMAGES = _CN.PATHS.MASKS_PATH +"/combined_masks_images"


    _CN.PATHS.CUT_MASKS = _CN.PATHS.MASKS_PATH + "/cut_masks"

    _CN.PATHS.SCALED_FRAME_PATH_ROOT = _CN.PATHS.MASKS_PATH+ "/" +"scaled_masks"
    _CN.FRAME_SCALING.DO_SCALE = False
    _CN.FRAME_SCALING.DESIRED_SIZE = (1600, 900)
    _CN.FRAME_SCALING.DESIRED_FACTOR = 4
    _CN.FRAME_SCALING.INTERPOLATION = cv2.INTER_AREA

    _CN.SAMPLING.CUT_TO_MASK = True

    _CN.PATHS.SAMPLING_ROOT_PATH = _CN.PATHS.OUTPUT_ROOT_PATH + "/" + "sub_sampled"
    _CN.SAMPLING.SAMPLING_STRATEGY = "fixed" #"nth"
    _CN.SAMPLING.SAMPLING_FREQUENCY = 3
    _CN.SAMPLING.NUMBER_SAMPLES = 20

    _CN.SAMPLING.BLUR_SCORE = CN()
    _CN.SAMPLING.BLUR_SCORE.DO_BLUR_SCORE = True


    _CN.PATHS.RECONSTRUCTION_ROOT_PATH  = _CN.PATHS.OUTPUT_ROOT_PATH + "/" +"reconstructions"

    _CN.PATHS.MAST3R_ROOT_PATH = _CN.PATHS.RECONSTRUCTION_ROOT_PATH + "/" +"mast3r"
    _CN.MAST3R.RANDOM_SEED = 42
    _CN.MAST3R.SCENEGRAPH_TYPE = "complete"

    _CN.POSTPROCESSING = CN()

    _CN.POSTPROCESSING.DENOISING = CN()
    _CN.POSTPROCESSING.DENOISING.DO_COLOR_FILTER    = False
    _CN.POSTPROCESSING.DENOISING.COLOR_FILTER_MIN   = 1.0
    _CN.POSTPROCESSING.DENOISING.COLOR_FILTER_MAX   = 253.0
    _CN.POSTPROCESSING.DENOISING.DO_NOISE_FILTER    = False
    _CN.POSTPROCESSING.DENOISING.NOISE_NB_NEIGHBORS = 10
    _CN.POSTPROCESSING.DENOISING.NOISE_STD_RATIO    = 1.5
    _CN.POSTPROCESSING.DENOISING.DO_SOR             = False
    _CN.POSTPROCESSING.DENOISING.SOR_NB_NEIGHBORS   = 20
    _CN.POSTPROCESSING.DENOISING.SOR_STD_RATIO      = 2.0
    _CN.POSTPROCESSING.DENOISING.SOR_PASSES         = 3

    _CN.POSTPROCESSING.MLS = CN()
    _CN.POSTPROCESSING.MLS.DO_MLS                   = False
    _CN.POSTPROCESSING.MLS.SEARCH_RADIUS            = 0.7

    _CN.POSTPROCESSING.POISSON_MESHER = CN()
    _CN.POSTPROCESSING.POISSON_MESHER.DO_POISSON      = False
    _CN.POSTPROCESSING.POISSON_MESHER.DEPTH           = 9
    _CN.POSTPROCESSING.POISSON_MESHER.DENSITY_QUANTILE = 0.1

    _CN.POSTPROCESSING.NKSR = CN()
    _CN.POSTPROCESSING.NKSR.DETAIL_LEVEL = None  # None = automatic; float 0-1 for explicit control
    _CN.POSTPROCESSING.NKSR.MISE_ITER = 1         # dual-mesh extraction iterations
    _CN.POSTPROCESSING.NKSR.OUTPUT_NAME = "mesh.obj"

    if experiment_config is not None:
        _CN.merge_from_file(experiment_config)
    return lower_config(_CN)
