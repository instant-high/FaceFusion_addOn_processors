from argparse import ArgumentParser
from typing import List, TypedDict

import cv2
import numpy

import facefusion.processors.core as processors
from facefusion import content_analyser, process_manager, state_manager
from facefusion.types import ApplyStateItem, Args, DownloadScope, ExecutionProvider, InferencePool, ModelOptions, ModelSet, ProcessMode, VisionFrame
from facefusion.vision import read_image, read_static_image, write_image
from facefusion.program_helper import find_argument_group

pencilInputs = TypedDict('pencilInputs',
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
        '--frame-pencil',
        action='store_true',
        help='Apply the passive pencil sketch effect to each frame.'
    )

def apply_args(args: Args, apply_state_item: ApplyStateItem) -> None:
    apply_state_item('pencil', args.get('pencil'))
	
def pre_check() -> bool:
    return True

def pre_process(mode: ProcessMode) -> bool:
    return True

def post_process() -> None:
    read_static_image.cache_clear()
    if state_manager.get_item('video_memory_strategy') == 'strict':
        content_analyser.clear_inference_pool()

def pencil(temp_vision_frame: numpy.ndarray) -> numpy.ndarray:
    kernel_sharpening = numpy.array([[-1,-1,-1],[-1, 9,-1],[-1,-1,-1]])
    sharpened = cv2.filter2D(temp_vision_frame,-1,kernel_sharpening)
    gray = cv2.cvtColor(sharpened , cv2.COLOR_BGR2GRAY)
    inv = 255-gray
    gaussgray = cv2.GaussianBlur(inv,ksize=(15,15),sigmaX=0,sigmaY=0)
    def dodgeV2(image,mask):
        return cv2.divide(image,255-mask,scale=256)
    pencil_img = dodgeV2(gray,gaussgray)
    pencil_img = cv2.cvtColor(pencil_img , cv2.COLOR_GRAY2RGB)
      
    return pencil_img

def process_frame(inputs : pencilInputs) -> VisionFrame:
		vision_frame = inputs.get('temp_vision_frame')
		if vision_frame is None:
			vision_frame = inputs.get('target_vision_frame')
		pencil_frame = pencil(vision_frame)

		return pencil_frame  
