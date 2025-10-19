from argparse import ArgumentParser
from typing import List, TypedDict

import cv2
import numpy

import facefusion.processors.core as processors
from facefusion import content_analyser, process_manager, state_manager
from facefusion.types import ApplyStateItem, Args, DownloadScope, ExecutionProvider, InferencePool, ModelOptions, ModelSet, ProcessMode, VisionFrame
from facefusion.vision import read_image, read_static_image, write_image
from facefusion.program_helper import find_argument_group

cartoonInputs = TypedDict('cartoonInputs',
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
        '--frame-cartoon',
        action='store_true',
        help='Apply the passive cartoon effect to each frame.'
    )

def apply_args(args: Args, apply_state_item: ApplyStateItem) -> None:
    apply_state_item('cartoonizer', args.get('cartoonizer'))
	
def pre_check() -> bool:
    return True

def pre_process(mode: ProcessMode) -> bool:
    return True

def post_process() -> None:
    read_static_image.cache_clear()
    if state_manager.get_item('video_memory_strategy') == 'strict':
        content_analyser.clear_inference_pool()

def Cartoon(temp_vision_frame: numpy.ndarray) -> numpy.ndarray:
    smoothened_image = cv2.bilateralFilter(temp_vision_frame, d=9, sigmaColor=75, sigmaSpace=75)
    gray_image = cv2.cvtColor(smoothened_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    cartoon = cv2.bitwise_and(smoothened_image, smoothened_image, mask=edges)
       
    return cartoon

def __Cartoon(temp_vision_frame: numpy.ndarray, color_levels: int = 8) -> numpy.ndarray:
    smooth = cv2.bilateralFilter(temp_vision_frame, d=9, sigmaColor=75, sigmaSpace=75)
    
    div = 256 // color_levels
    reduced = smooth // div * div

    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
        9, 9
    )

    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(reduced, edges_colored)

    return cartoon
    
def process_frame(inputs : cartoonInputs) -> VisionFrame:
		vision_frame = inputs.get('temp_vision_frame')
		if vision_frame is None:
			vision_frame = inputs.get('target_vision_frame')
		cartoon_frame = Cartoon(vision_frame)

		return cartoon_frame  
		
