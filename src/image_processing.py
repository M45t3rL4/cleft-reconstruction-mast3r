import concurrent
from pathlib import Path
import cv2
from tqdm import tqdm
from deepdiff import DeepDiff
from src.utils import new_numbered_folder
from src.config import get_all_configs
import yaml
import os
from src.config import print_conf
import logging

logger = logging.getLogger("CleftReconstruction")

def resize_images_exist(new_config, path):
    configs = get_all_configs(path)
    for parent_path, config in configs:
        diff = DeepDiff(new_config, config)
        if len(diff.affected_paths) == 0:
            return parent_path
    return None


def helper_resize_images(input_image, output_path, desired_factor, desired_size, interpolation):
    if desired_factor is not None:
        cur_img = cv2.imread(str(input_image))

        new_size = (int(cur_img.shape[1] * desired_factor), int(cur_img.shape[0] * desired_factor))
        resized_img = cv2.resize(cur_img, new_size, interpolation=interpolation)
        cv2.imwrite(output_path + f"/{input_image.name}", resized_img)
    else:
        cur_img = cv2.imread(str(input_image))
        if desired_size[0] is not None and desired_size[1] is not None:
            new_size = desired_size
        elif desired_size[0] is None:
            new_size = (cur_img.shape[1], desired_size[1])
        else:
            new_size = (desired_size[0], cur_img.shape[0])
        resized_img = cv2.resize(cur_img, new_size, interpolation=interpolation)
        cv2.imwrite(output_path + f"/{input_image.name}", resized_img)


def resize_images(resize_frames_conf, root_output_path, no_threads=12):
    cur_output_path = new_numbered_folder(root_output_path, "scaled_masks")
    desired_size = resize_frames_conf['desired_size']
    desired_factor = resize_frames_conf['desired_factor']
    interpolation = resize_frames_conf['interpolation']

    if desired_size[0] is None and desired_size[1] is None and desired_factor is None:
        logger.warning(f"\t Aborting scaling as no factor or size is given.")
        return None
    if desired_size is not None and desired_factor is not None:
        logger.warning(f"\t Both size and factor are given, resizing with factor {desired_factor}")

    cur_frames = list(Path(resize_frames_conf['combined_masks_input_path']).glob('*.png'))

    executor = concurrent.futures.ProcessPoolExecutor(no_threads)
    futures = []
    with tqdm(total=len(cur_frames), desc="Resizing Images") as progress:
        for frame in cur_frames:
            future = executor.submit(helper_resize_images, frame, cur_output_path, desired_factor, desired_size,
                                     interpolation)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)
        concurrent.futures.wait(futures)

    with open(f"{cur_output_path}/cur_config.yaml", 'w') as file:
        documents = yaml.dump(resize_frames_conf, file, default_flow_style=False)
    return cur_output_path


def helper_calculate_blur_scores(image):
    cur_mask = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    cur_score = cv2.Laplacian(cur_mask, cv2.CV_64F)
    cur_score = cur_score.flatten()
    cur_score = cur_score[cur_score != 0]
    if len(cur_score) == 0:
        cur_score = 0
    else:
        cur_score = cur_score.var()
    return image.name, cur_score


def calculate_blur_scores(blur_score_conf, no_threads=12):
    score_dic = {}
    input_mask_path = blur_score_conf['reference_mask_path']

    blur_file_path = os.path.join(input_mask_path, "blur_scores.yaml")

    cur_frames = list(Path(input_mask_path).glob('*.png'))
    score_dics = list(Path(input_mask_path).glob('blur_scores.yaml'))
    for dic in score_dics:
        with open(dic, 'r') as file:
            score_dic = yaml.load(file, Loader=yaml.Loader)

        if 'scores' in score_dic and 'conf' in score_dic and len(score_dic['scores']) == len(cur_frames) and len(
                DeepDiff(score_dic['conf'], blur_score_conf).affected_paths) == 0:
            logger.info(f"SKIPPING BLUR CALCULATION\n"
                        f"\t Exists at {blur_file_path}"
                        f"\t With config: {print_conf(blur_score_conf)}")
            return score_dic

    score_dic = {'conf': blur_score_conf, 'scores': {}}

    executor = concurrent.futures.ProcessPoolExecutor(no_threads)
    futures = []
    with tqdm(total=len(cur_frames), desc="Calculate Blur Scores") as progress:
        for frame in cur_frames:
            future = executor.submit(helper_calculate_blur_scores, frame)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)
        for future in futures:
            result_name, result_score = future.result()
            score_dic['scores'][result_name] = result_score

    with open(blur_file_path, 'w') as file:
        documents = yaml.dump(score_dic, file, default_flow_style=False)

    return score_dic