import concurrent
from pathlib import Path
import cv2
from tqdm import tqdm
from deepdiff import DeepDiff
from src.utils import new_numbered_folder
from src.config import get_all_configs
import yaml
import gc


def extraced_frames_exist(new_config, path):
    configs = get_all_configs(path)
    for parent_path, config in configs:
        diff = DeepDiff(new_config, config)
        if len(diff.affected_paths) == 0:
            no_frames = len(list(Path(parent_path).glob('*.png')))
            return no_frames, parent_path
    return None, None


def helper_extract_frames(image, i, output_path, mirror):
    if mirror:
        image = cv2.flip(image, -1)
    cv2.imwrite(output_path + '/frame%04d.png' % i, image)


def extract_frames(video_path, frame_root_path, extract_frames_conf, no_threads=8):
    video = cv2.VideoCapture(video_path)
    no_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path = new_numbered_folder(frame_root_path, 'frames')
    # for i in trange(no_frames, desc="Frame Extraction"):
    #     success, image = video.read()
    #     helper_extract_frames(image, i, output_path, extract_frames_conf['mirror_images'])

    # Limited by IO, multithreading speeds roughly x4, also futures are large apparently because they store the image
    # and run out of RAM
    futures = []
    executor = concurrent.futures.ProcessPoolExecutor(no_threads)
    with tqdm(total=no_frames, desc="Frame Extraction") as progress:
        for i in range(no_frames):
            if i % no_threads == 0:
                concurrent.futures.wait(futures)
                futures.clear()
                gc.collect()
            success, image = video.read()
            future = executor.submit(helper_extract_frames, image, i, output_path, extract_frames_conf['mirror_images'])
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        concurrent.futures.wait(futures)

    with open(f"{output_path}/cur_config.yaml", 'w') as file:
        documents = yaml.dump(extract_frames_conf, file, default_flow_style=False)
    return no_frames, output_path