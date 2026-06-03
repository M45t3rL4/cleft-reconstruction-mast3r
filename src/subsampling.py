from pathlib import Path
from deepdiff import DeepDiff
from src.utils import new_numbered_folder
from src.config import get_all_configs
import yaml
import os
import logging
from src.image_processing import calculate_blur_scores
import math
from tqdm import trange
logger = logging.getLogger("CleftReconstruction")


def subsample_exists(new_config, path):
    configs = get_all_configs(path, config_name='subsample.yaml')
    for parent_path, config in configs:
        diff = DeepDiff(new_config, config['sample_conf'])
        if len(diff.affected_paths) == 0:
            if len(list(Path(parent_path).glob('*.png'))) == 0:
                continue
            else:
                return config['images'], parent_path
    return None, None


def subsample_frames(input_path, sample_root_path, sampling_conf):
    cur_output = new_numbered_folder(sample_root_path, 'subsample')
    if sampling_conf["blur_score"]["do_blur_score"]:
        score_dic = calculate_blur_scores(sampling_conf["blur_score"])

    sample_dic = {'images': [], 'sample_conf': sampling_conf}

    cur_frames = list(Path(input_path).glob('*.png'))
    if sampling_conf['sampling_strategy'] == 'nth':
        sampling_frequency = sampling_conf['sampling_frequency']
    elif sampling_conf['sampling_strategy'] == "fixed":
        no_frames = len(cur_frames)
        if no_frames <= sampling_conf['number_samples'] +1:
            return None, None
        sampling_frequency = math.floor(max(1, no_frames / sampling_conf['number_samples']))
    else:
        logger.warning(
            f"Sampling strategy {sampling_conf['sampling_strategy']} is not implemented yet, only \'nth\' is available")
        return None

    max_score = 0
    max_index = 0
    for i in trange(len(cur_frames), desc="Subsample frames"):
        if not sampling_conf["blur_score"]["do_blur_score"]:
            if int(cur_frames[i].name[-8:-4]) % sampling_frequency == 1:
                sample_dic['images'].append(str(cur_frames[i]))
        else:
            if score_dic['scores'][cur_frames[i].name] > max_score:
                max_score = score_dic['scores'][cur_frames[i].name]
                max_index = i
            if int(cur_frames[i].name[-8:-4]) % sampling_frequency == 1:
                sample_dic['images'].append(str(cur_frames[max_index]))
                max_score = 0
                max_index = i + 1
    for image in sample_dic['images']:
        os.symlink(os.path.abspath(image), os.path.abspath(os.path.join(cur_output, Path(image).name)))
    with open(cur_output + '/subsample.yaml', 'w') as file:
        documents = yaml.dump(sample_dic, file, default_flow_style=False)

    return sample_dic['images'], cur_output

