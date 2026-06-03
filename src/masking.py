import os

from deepdiff import DeepDiff
import logging
from pathlib import Path
from src.config import get_all_configs
import cv2
import yaml
import numpy as np
import shutil
import concurrent
from tqdm import tqdm, trange
from src.utils import new_numbered_folder
from thirdparty.Cutie.interactive_demo import start_app
logger = logging.getLogger("CleftReconstruction")

def generate_masks(video_path, mask_path, target_number):
    logger.info(f"STARTING MASK EXTRACTION:")
    existing_masks = len(list(Path(mask_path).glob('*.png')))
    if existing_masks == target_number:
        logger.info(f"SKIPPING: Mask extraction, already exists at {mask_path}")
        return True
    else:
        logger.info(f"Mask extraction, found {existing_masks} but needs {target_number}")
        #TODO give path as parameter
        temp_path = '../output/temp'
        os.makedirs(temp_path,exist_ok=True)
        start_app(video_path, temp_path)
        #TODO get returncode
        # if result.returncode != 0:
        #     logger.info(f"Masks subprocess failed with return code {result.returncode}")
        #     return False
        for filename in os.listdir(f'{temp_path}/masks/'):
            shutil.copy(f'{temp_path}/masks/{filename}', f'{mask_path}/')
        shutil.rmtree(f'{temp_path}/images',        ignore_errors=True)
        shutil.rmtree(f'{temp_path}/masks',         ignore_errors=True)
        shutil.rmtree(f'{temp_path}/soft_masks',    ignore_errors=True)
        shutil.rmtree(f'{temp_path}/visualization', ignore_errors=True)
        shutil.rmtree(f'{temp_path}/binary_masks',  ignore_errors=True)
        existing_masks = len(list(Path(mask_path).glob('*.png')))
        if existing_masks != target_number:
            logger.info(f"Exited application too early, not enough masks created {mask_path}")
            return False
        logger.info(f"Masks located at {mask_path}")
        return True


def rem_highlights_exist(new_config, path):
    configs = get_all_configs(path)
    for parent_path, config in configs:
        diff = DeepDiff(new_config, config)
        if len(diff.affected_paths) == 0:
            return parent_path
    return None


def helper_rem_highlights(file, threshold, cur_output):
    cur_img = cv2.imread(str(file))
    gray_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
    ret, threshold_img = cv2.threshold(gray_img, threshold, 255,
                                       cv2.THRESH_BINARY_INV)
    cv2.imwrite(cur_output + f"/{file.name}", threshold_img)


# TODO check if all highlight files are there
def remove_highlights(input_path, output_path, highlight_removal_conf, no_threads=12):
    cur_output = new_numbered_folder(output_path, 'highlight_masks')
    input_frames = list(Path(input_path).glob('*.png'))

    executor = concurrent.futures.ProcessPoolExecutor(no_threads)
    futures = []
    with tqdm(total=len(input_frames), desc="Removing Highlights") as progress:
        for frame in input_frames:
            future = executor.submit(helper_rem_highlights, frame, highlight_removal_conf['threshold'], cur_output)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        concurrent.futures.wait(futures)

    with open(f"{cur_output}/cur_config.yaml", 'w') as file:
        documents = yaml.dump(highlight_removal_conf, file, default_flow_style=False)

    return cur_output


#TODO Check if masks exists in folders too
def combined_masks_exist(new_config, combined_masks_path, combined_masks_images_path):
    path1 = None
    path2 = None
    configs = get_all_configs(combined_masks_path)
    for parent_path, config in configs:
        diff = DeepDiff(new_config, config)
        if len(diff.affected_paths) == 0:
            path1 = parent_path

    configs = get_all_configs(combined_masks_images_path)
    for parent_path, config in configs:
        diff = DeepDiff(new_config, config)
        if len(diff.affected_paths) == 0:
            path2 = parent_path

    if path1 is not None and path2 is not None:
        return path1, path2
    return None, None


def helper_combine(file, highlight_input, mirror_input, input_frames, output_path_masks_images, output_path_masks,
                   mirror):
    # file = cur_frames[i]
    mirror_mask_no = int(file.name[-8:-4])
    mirror_mask_name = f'{mirror_mask_no + 1:07d}' + ".png"  # 05d wihtout _1
    mask1 = cv2.imread(highlight_input + f"/{file.name}", cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread(mirror_input + f"/{mirror_mask_name}", cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(input_frames + f"/{file.name}")

    mask1 = cv2.resize(mask1, (image.shape[1], image.shape[0]))
    mask2 = cv2.resize(mask2, (image.shape[1], image.shape[0]))
    if mirror:
        mask2 = cv2.flip(mask2, -1)
    combined_mask = cv2.bitwise_and(mask1, mask1, mask=mask2)
    res1 = cv2.bitwise_and(image, image, mask=mask1)
    res2 = cv2.bitwise_and(res1, res1, mask=mask2)
    cv2.imwrite(output_path_masks_images + f"/{file.name}", res2)
    cv2.imwrite(output_path_masks + f"/{file.name}", combined_mask)

def combine_masks(combine_masks_config, output_path_combined_masks, output_path_combined_masks_images, no_threads=12):
    output_path_masks = new_numbered_folder(output_path_combined_masks, 'combined_masks')
    output_path_masks_images = new_numbered_folder(output_path_combined_masks_images, 'combined_masks_images')

    cur_frames = list(Path(combine_masks_config['reference_frame_path']).glob('*.png'))
    # TODO better handover
    with open(combine_masks_config['reference_frame_path'] + "/cur_config.yaml", 'r') as file:
        test = yaml.load(file, Loader=yaml.Loader)
        mirror = test['mirror_images']

    executor = concurrent.futures.ProcessPoolExecutor(no_threads)
    futures = []
    with tqdm(total=len(cur_frames), desc="Combining Masks") as progress:
        for frame in cur_frames:
            future = executor.submit(helper_combine, frame, combine_masks_config['reference_highlight_mask_path'],
                                     combine_masks_config['mask_mirror_path'],
                                     combine_masks_config['reference_frame_path'],
                                     output_path_masks_images, output_path_masks, mirror)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)
        concurrent.futures.wait(futures)

    with open(f"{output_path_masks}/cur_config.yaml", 'w') as file:
        documents = yaml.dump(combine_masks_config, file, default_flow_style=False)
    with open(f"{output_path_masks_images}/cur_config.yaml", 'w') as file:
        documents = yaml.dump(combine_masks_config, file, default_flow_style=False)
    return output_path_masks, output_path_masks_images


def cut_to_mask(input_path, mask_path, output_path):
    logger.info("Cut images only to relevant Mask parts")
    current_frames = sorted(Path(input_path).glob('*.png'), key=lambda p: p.name)
    current_masks  = {p.name: p for p in Path(mask_path).glob('*.png')}
    for i in trange(len(current_frames), desc="Cut To Masks"):
        img = cv2.imread(str(current_frames[i]))

        mask_file = current_masks.get(current_frames[i].name)
        if mask_file is None:
            logger.warning(f"No mask found for {current_frames[i].name}, skipping")
            continue
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0 :
        # #find maximum contour and draw
            cmax = max(contours, key=cv2.contourArea)
        # epsilon = 0.002 * cv2.arcLength(cmax, True)
        # approx = cv2.approxPolyDP(cmax, epsilon, True)

            width, height = mask.shape
            new_mask = np.zeros([width, height, 1], dtype=np.uint8)
            cv2.fillPoly(new_mask, pts=[cmax], color=255)

            x, y, w, h = cv2.boundingRect(new_mask)

            new_img = cv2.bitwise_and(img, img, mask=new_mask)
        # combine image with mask and cut to rectangle
            cropped_image = new_img[y:y + h, x:x + w]
            cv2.imwrite(str(output_path) + f"/{current_frames[i].name}", cropped_image)
    logger.info(f"Cut masks found at{output_path}")

