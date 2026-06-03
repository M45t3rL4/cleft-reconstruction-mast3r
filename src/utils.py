import os
from pathlib import Path
import logging
logger = logging.getLogger("CleftReconstruction")

def new_numbered_folder(path, name):
    counter = 0
    new_name = name + str(counter)
    while os.path.isdir(os.path.join(path, new_name)):
        counter += 1
        new_name = name + str(counter)
    os.makedirs(os.path.join(path, new_name))
    return os.path.join(path, new_name)


def new_numbered_file(path, name, ending):
    counter = 0
    new_name = name + str(counter)
    while os.path.isfile(os.path.join(path, new_name + ending)):
        counter += 1
        new_name = name + str(counter)
    return os.path.join(path, new_name + ending)

def get_files_recursive(file_path: Path, filetypes, include_list=None, exclude_list=None):
    cur_list = []
    if os.path.isdir(file_path):
        for ft in filetypes:
            for e in list(Path(file_path).rglob(f'*.{ft}')):
                cur_list.append(str(e))
    elif os.path.isfile(file_path) and Path(file_path).suffix in filetypes:
        cur_list.append(str(file_path))

    if len(cur_list) == 0:
        logger.warning(f"In the path {file_path} no video files of type {filetypes} have been found.")
    if include_list is not None:
        cur_list = list(set(cur_list) | set(include_list))
    if exclude_list is not None:
        cur_list = list(set(cur_list) - set(exclude_list))

    logger.info(f"The current video list contains {len(cur_list)} videos.")
    return cur_list


def change_cwd():
    logger.info('Changing Current Directory\n')
    while not os.getcwd().endswith("cleft_reconstruction"):
        if 3 > len(os.getcwd().split("/")) > 0 or 4 > len(os.getcwd().split('\\')) > 1:
            print(
                f"Can not find working directory named" +
                f" \'cleft_reconstruction\' in current or parent directory. {os.getcwd()}")
            exit()
        os.chdir('../')
    os.environ['PATH'] += os.pathsep + "/usr/local/bin/colmap"
    logger.info(f'Working directory is now: {os.getcwd()}\n')