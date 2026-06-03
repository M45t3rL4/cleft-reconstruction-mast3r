import concurrent
import os

from src.image_processing import resize_images_exist, resize_images
from src.masking import cut_to_mask, generate_masks, rem_highlights_exist, remove_highlights, combined_masks_exist, \
    combine_masks
from src.subsampling import subsample_exists, subsample_frames
from src.utils import get_files_recursive
from src.config import *
from pathlib import Path
import logging
from src.video_utils import extract_frames, extraced_frames_exist
from src.reconstructions import mast3r_reconstruction
from src.mesh_reconstruction import nksr_reconstruction
from src.mesh_processing import denoise_point_cloud, mls_and_poisson

VIDEO_FILE_TYPES= ['mp4', 'MOV']
# GOOGLE ANSI ESCAPE CODE FOR COLORS
logging.basicConfig(level=logging.INFO,
                    format='[\x1b[36m %(asctime)s %(name)s %(levelname)s] \x1b[32m %(message)s \x1b[39m')
logger = logging.getLogger("CleftReconstruction")



if __name__ == "__main__":
    # change_cwd()
    default_conf = generate_config(None, None, None)
    input_dir = default_conf['paths']['input_video_dir']
    if input_dir == 'your path here':
        raise ValueError("INPUT_VIDEO_DIR is not set — update _CN.PATHS.INPUT_VIDEO_DIR in src/config.py")
    videos = get_files_recursive(input_dir, filetypes=VIDEO_FILE_TYPES)
    counter = 0
    successful = 0
    for vid in videos:
        try:
            counter += 1
            # ----------------------------- PREPARATION -----------------------------
            logger.info(
                f"\n ------------------------------ Starting Video Reconstruction {counter}/{len(videos)}/{successful} ------------------------------------- "
                f"\n Video: {Path(vid).name} "
                f"\n Path: {Path(vid)}"
                f"\n")
            latest_output_path = None
            name = Path(vid).stem
            conf = generate_config(str(vid), name, None)
            for k in conf['paths']:
                if k != 'input_video_path':
                    os.makedirs(conf['paths'][k], exist_ok=True)
                    # if k == 'reconstruction_root_path' or k == 'sampling_root_path':
                    #     print("delete",conf['paths'][k] )
                    #     shutil.rmtree(conf['paths'][k])

            # ----------------------------- FRAME EXTRACTION -----------------------------
            no_frames, cur_frame_path = extraced_frames_exist(conf['video_conversion'], conf['paths']['root_frame_path'])
            if cur_frame_path is None:
                logger.info(f"STARTING VIDEO CONVERSION:\n"
                            f"\t Video: {conf['paths']['input_video_path']} \n"
                            f"\t Config: {print_conf(conf['video_conversion'])}")
                no_frames, cur_frame_path = extract_frames(conf['paths']['input_video_path'],
                                                           conf['paths']['root_frame_path'], conf['video_conversion'])
                logger.info("Finished video conversion")
            else:
                logger.info(f"SKIPPING CONVERSION: \n"
                            f"\t Given frames exist at: {cur_frame_path} \n"
                            f"\t With config:  {print_conf(conf['video_conversion'])}")

            if not generate_masks(conf['paths']['input_video_path'], conf['paths']['mask_mirror_path'], no_frames):
                logger.info("Mask creation failed, skipping")
                continue


            # -----------------------------HIGHLIGHT REMOVAL -----------------------------
            conf['highlight_removal']['reference_frame_path'] = str(cur_frame_path)
            highlight_masks_path = rem_highlights_exist(conf['highlight_removal'], conf['paths']['highlight_masks_root'])
            if highlight_masks_path is None:
                logger.info(f"STARTING HIGHLIGHT MASKING:\n"
                            f"\t Config: {print_conf(conf['highlight_removal'])}")
                highlight_masks_path = remove_highlights(cur_frame_path, conf['paths']['highlight_masks_root'],
                                                         conf['highlight_removal'])
            else:
                logger.info(f"SKIPPING HIGHLIGHT MASKS: \n"
                            f"\t Given masks exist at: {highlight_masks_path} \n"
                            f"\t With config:  {print_conf(conf['highlight_removal'])}")


            # ----------------------------- MASK COMBINATION -----------------------------
            conf['mask_combination'] = {}
            conf['mask_combination']['reference_frame_path'] = str(cur_frame_path)
            conf['mask_combination']['reference_highlight_mask_path'] = str(highlight_masks_path)
            conf['mask_combination']['mask_mirror_path'] = conf['paths']['mask_mirror_path']
            combined_mask_path, combined_mask_image_path = combined_masks_exist(conf['mask_combination'],
                                                                                conf['paths']['combined_masks'],
                                                                                conf['paths']['combined_masks_images'])
            if combined_mask_path is None or combined_mask_image_path is None:
                logger.info(f"STARTING MASK COMBINATION:\n"
                            f"\t Config: {print_conf(conf['mask_combination'])}")
                combined_mask_path, combined_mask_image_path = combine_masks(conf['mask_combination'],
                                                                             conf['paths']['combined_masks'],
                                                                             conf['paths']['combined_masks_images'])
            else:
                logger.info(f"SKIPPING MASK COMBINATION:\n"
                            f"\t Combined masks exist at: {combined_mask_path} \n"
                            f"\t Combined masks and images exist at: {combined_mask_image_path} \n"
                            f"\t Config: {print_conf(conf['mask_combination'])}")
            # ----------------------------- FRAME SCALING -----------------------------
            # Note that most methods already include scaling later on
            if conf['frame_scaling']['do_scale']:
                conf['frame_scaling']['combined_masks_input_path'] = str(combined_mask_image_path)
                scaled_frame_output = resize_images_exist(conf['frame_scaling'], conf['paths']['scaled_frame_path_root'])
                if scaled_frame_output is None:
                    logger.info(f"STARTING MASK RESIZING:\n"
                                f"\t Config: {print_conf(conf['frame_scaling'])}")
                    scaled_frame_output = resize_images(conf['frame_scaling'], conf['paths']['scaled_frame_path_root'])
                    logger.info(f"\t Resized Path: {scaled_frame_output}")
                else:
                    logger.info(f"RESIZED MASKS EXIST:\n"
                                f"\t Resized Path: {scaled_frame_output} \n"
                                f"\t Config: {print_conf(conf['frame_scaling'])}")
                latest_output_path = scaled_frame_output
            else:
                logger.info(f"SKIPPING MASK SCALING\n")
                latest_output_path = combined_mask_image_path

            #TODO dont double cut
            if conf['sampling']['cut_to_mask']:
                # cut_to_mask(combined_mask_image_path, combined_mask_path, conf['paths']['cut_masks']) #TODO unccoment after check
                latest_output_path = conf['paths']['cut_masks']

            # ----------------------------- SUBSAMPLING -----------------------------
            # Always calculate the score for the masks, as it should scale the same to different sizes
            conf['sampling']["blur_score"]['reference_mask_path'] = str(combined_mask_image_path)
            subsampled_images, subsampled_path = subsample_exists(conf['sampling'], conf['paths']['sampling_root_path'])
            if subsampled_images is None or subsampled_path is None:
                logger.info(f"STARTING SUBSAMPLING:\n"
                            f"\t Config: {print_conf(conf['sampling'])}")
                subsampled_images, subsampled_path = subsample_frames(latest_output_path,
                                                                      conf['paths']['sampling_root_path'], conf['sampling'])
                if subsampled_images is None or subsampled_path is None:
                    #TODO give error message from previous function call subsample_frames
                    logger.info(f"\t Requested more subsamples than frames exist")
                    continue

                logger.info(f"\t Subsamples found at: {subsampled_path}")
            else:
                logger.info(f"SKIPPING SUBSAMPLING:\n"
                            f"\t Exists at {subsampled_path} \n"
                            f"\t Config: {print_conf(conf['sampling'])}")

            # ----------------------------- MAST3R -----------------------------
            mast3r_reconstruction(subsampled_path, conf['paths']['mast3r_root_path'], conf['mast3r'])

            # ----------------------------- DENOISING + MESH RECONSTRUCTION -----------------------------
            ply_path = Path(conf['paths']['mast3r_root_path']) / 'sparse.ply'
            denoised_ply = denoise_point_cloud(ply_path, conf['paths']['mast3r_root_path'],
                                               conf['postprocessing']['denoising'])
            nksr_reconstruction(denoised_ply, conf['paths']['mast3r_root_path'], conf['postprocessing']['nksr'])
            mls_and_poisson(denoised_ply, conf['paths']['mast3r_root_path'],
                            conf['postprocessing']['mls'], conf['postprocessing']['poisson_mesher'])
        except Exception as e:
            logger.error(f"Failed processing {Path(vid).name}: {e}", exc_info=True)
            continue

