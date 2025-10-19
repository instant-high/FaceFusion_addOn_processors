from argparse import ArgumentParser
from typing import List, TypedDict

import cv2
import numpy

import facefusion.processors.core as processors
from facefusion import content_analyser, process_manager, state_manager
from facefusion.types import ApplyStateItem, Args, DownloadScope, ExecutionProvider, InferencePool, ModelOptions, ModelSet, ProcessMode, VisionFrame
from facefusion.vision import read_image, read_static_image, write_image
from facefusion.program_helper import find_argument_group

awbInputs = TypedDict('awbInputs',
{
    'target_vision_frame' : VisionFrame
})

def get_inference_pool() -> InferencePool:
    pass

def clear_inference_pool() -> None:
    pass

def register_args(program: ArgumentParser) -> None:
    group_processors = find_argument_group(program, 'processors')
    if group_processors:
        group_processors.add_argument(
        '--frame-awb',
        action='store_true',
        help='Apply the passive auto_white_balance effect to each frame.'
    )

def apply_args(args: Args, apply_state_item: ApplyStateItem) -> None:
    apply_state_item('auto_white_balance', args.get('auto_white_balance'))
	
def pre_check() -> bool:
    return True

def pre_process(mode: ProcessMode) -> bool:
    return True

def post_process() -> None:
    read_static_image.cache_clear()
    if state_manager.get_item('video_memory_strategy') == 'strict':
        content_analyser.clear_inference_pool()

def AutoWhiteBalance(temp_vision_frame: numpy.ndarray) -> numpy.ndarray:
    # some more color profiles:
    # "A"   : 0.44757, 0.40745 
    # "B"   : 0.34842, 0.35161
    # "C"   : 0.31006, 0.31616
    # "D65" : 0.31271, 0.32902
    # "D93" : 0.28315, 0.29711
    # "E"   : 0.33333, 0.33333
    
    # eg. fixed D65 color values:
    cie_x, cie_y = 0.31271, 0.32902
    cie_Y = 250
    cie_X = cie_Y * cie_x / cie_y
    cie_Z = cie_Y * (1 - cie_x - cie_y) / cie_y
    temperature_color = numpy.array([cie_X, cie_Y, cie_Z])

    temp_vision_frame = cv2.cvtColor(temp_vision_frame, cv2.COLOR_BGR2XYZ).astype(numpy.float32)

    num_max = int(0.05 * (temp_vision_frame.shape[0] * temp_vision_frame.shape[1]))
    top_xyz = numpy.sort(temp_vision_frame.reshape(-1, 3), axis=0)[::-1][:num_max]
    max_xyz = numpy.mean(top_xyz, axis=0)

    K_xyz = temperature_color / max_xyz
    temp_vision_frame *= K_xyz

    return numpy.clip(cv2.cvtColor(temp_vision_frame, cv2.COLOR_XYZ2BGR), 0, 255).astype(numpy.uint8)

def process_frame(inputs : awbInputs) -> VisionFrame:
		vision_frame = inputs.get('temp_vision_frame')
		if vision_frame is None:
			vision_frame = inputs.get('target_vision_frame')
		awb_frame = AutoWhiteBalance(vision_frame)

		return awb_frame  
