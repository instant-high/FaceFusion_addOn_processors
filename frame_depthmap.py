from argparse import ArgumentParser
from functools import lru_cache
from typing import List, TypedDict

import cv2
import numpy

import facefusion.jobs.job_manager
import facefusion.jobs.job_store
import facefusion.processors.core as processors
from facefusion import config, content_analyser, inference_manager, logger, state_manager, video_manager, wording
from facefusion.common_helper import create_int_metavar, is_macos
from facefusion.download import conditional_download_hashes, conditional_download_sources, resolve_download_url
from facefusion.execution import has_execution_provider
from facefusion.filesystem import in_directory, is_image, is_video, resolve_relative_path, same_file_extension
from facefusion.processors import choices as processors_choices
from facefusion.program_helper import find_argument_group
from facefusion.thread_helper import thread_semaphore
from facefusion.types import ApplyStateItem, Args, DownloadScope, ExecutionProvider, InferencePool, ModelOptions, ModelSet, ProcessMode, VisionFrame
from facefusion.vision import blend_frame, read_static_image, read_static_video_frame, unpack_resolution

 
mode_3d = config.get_str_value('depth', '3D_mode', 'none').lower()

# 3D depth
shift_factor = 10

depthmapInputs = TypedDict('depthmapInputs',
{
	'target_vision_frame' : VisionFrame,
	'temp_vision_frame' : VisionFrame
})

@lru_cache(maxsize = None)
def create_static_model_set(download_scope: DownloadScope) -> ModelSet:
    return {
        'depth_anything_vits14': {
            'hashes': {
                'frame_depthmap': {
                    'url': resolve_download_url('models-3.0.0', 'depth_anything_vits14.hash'),
                    'path': resolve_relative_path('../.assets/models/depth_anything_vits14.hash')
                }
            },
            'sources': {
                'frame_depthmap': {
                    'url': resolve_download_url('models-3.0.0', 'depth_anything_vits14.onnx'),
                    'path': resolve_relative_path('../.assets/models/depth_anything_vits14.onnx')
                }
            },
            'type': 'small'
        }
    }



def get_inference_pool() -> InferencePool:
	model_names = [ state_manager.get_item('frame_depthmap_model') ]
	model_source_set = get_model_options().get('sources')

	return inference_manager.get_inference_pool(__name__, model_names, model_source_set)


def clear_inference_pool() -> None:
	model_names = [ state_manager.get_item('frame_depthmap_model') ]
	inference_manager.clear_inference_pool(__name__, model_names)


def get_model_options() -> ModelOptions:
	model_name = state_manager.get_item('frame_depthmap_model')
	return create_static_model_set('full').get(model_name)


def resolve_execution_providers() -> List[ExecutionProvider]:
	if is_macos() and has_execution_provider('coreml'):
		return [ 'cpu' ]
	return state_manager.get_item('execution_providers')
	
	
def register_args(program : ArgumentParser) -> None:
	group_processors = find_argument_group(program, 'processors')
	if group_processors:
		group_processors.add_argument('--frame-depthmap-model', help = wording.get('help.frame_depthmap_model'), default = config.get_str_value('processors', 'frame_depthmap_model', 'depth_anything_vits14'))
		facefusion.jobs.job_store.register_step_keys([ 'frame_depthmap_model'])


def apply_args(args : Args, apply_state_item : ApplyStateItem) -> None:
	apply_state_item('frame_depthmap_model', args.get('frame_depthmap_model'))


def pre_check() -> bool:
	model_hash_set = get_model_options().get('hashes')
	model_source_set = get_model_options().get('sources')

	return conditional_download_hashes(model_hash_set) and conditional_download_sources(model_source_set)


def pre_process(mode : ProcessMode) -> bool:
	if mode in [ 'output', 'preview' ] and not is_image(state_manager.get_item('target_path')) and not is_video(state_manager.get_item('target_path')):
		logger.error(wording.get('choose_image_or_video_target') + wording.get('exclamation_mark'), __name__)
		return False
	if mode == 'output' and not in_directory(state_manager.get_item('output_path')):
		logger.error(wording.get('specify_image_or_video_output') + wording.get('exclamation_mark'), __name__)
		return False
	if mode == 'output' and not same_file_extension(state_manager.get_item('target_path'), state_manager.get_item('output_path')):
		logger.error(wording.get('match_target_and_output_extension') + wording.get('exclamation_mark'), __name__)
		return False
	return True
	

def post_process() -> None:
	read_static_image.cache_clear()
	read_static_video_frame.cache_clear()
	video_manager.clear_video_pool()
	if state_manager.get_item('video_memory_strategy') in [ 'strict', 'moderate' ]:
		clear_inference_pool()
	if state_manager.get_item('video_memory_strategy') == 'strict':
		content_analyser.clear_inference_pool()


def depthmap_frame(temp_vision_frame: VisionFrame) -> VisionFrame:
	frame_depthmap_task = state_manager.get_item('frame_depthmap_task')
	depthmap_vision_frame = prepare_temp_frame(temp_vision_frame)

	depthmap_vision_frame = forward(depthmap_vision_frame)
	depthmap_vision_frame = cv2.resize(depthmap_vision_frame, (temp_vision_frame.shape[1], temp_vision_frame.shape[0]))
	
	if mode_3d == 'half_sbs':
		depth_scaled = (depthmap_vision_frame / numpy.max(depthmap_vision_frame)) * 255
		shift = ((255 - depth_scaled) / 255 * shift_factor).astype(int)
		stereogram = numpy.zeros_like(temp_vision_frame)
		shifted_indices = numpy.maximum(numpy.arange(temp_vision_frame.shape[1])[None, :] + shift[:, :], 0)
		shifted_indices = numpy.clip(shifted_indices, 0, temp_vision_frame.shape[1] - 1)
		stereogram[numpy.arange(temp_vision_frame.shape[0])[:, None], numpy.arange(temp_vision_frame.shape[1])] = temp_vision_frame[numpy.arange(temp_vision_frame.shape[0])[:, None], shifted_indices]
		stereogram = numpy.clip(stereogram, 0, 255)
		
		depthmap_vision_frame = numpy.concatenate((stereogram, temp_vision_frame), axis=1)


	if mode_3d == 'half_sbs_crosseye':
		depth_scaled = (depthmap_vision_frame / numpy.max(depthmap_vision_frame)) * 255
		shift = ((255 - depth_scaled) / 255 * shift_factor).astype(int)
		stereogram = numpy.zeros_like(temp_vision_frame)
		shifted_indices = numpy.maximum(numpy.arange(temp_vision_frame.shape[1])[None, :] + shift[:, :], 0)
		shifted_indices = numpy.clip(shifted_indices, 0, temp_vision_frame.shape[1] - 1)
		stereogram[numpy.arange(temp_vision_frame.shape[0])[:, None], numpy.arange(temp_vision_frame.shape[1])] = temp_vision_frame[numpy.arange(temp_vision_frame.shape[0])[:, None], shifted_indices]
		stereogram = numpy.clip(stereogram, 0, 255)

		depthmap_vision_frame = numpy.concatenate((temp_vision_frame, stereogram), axis=1)
						

	elif mode_3d == 'anaglyph':
		depth_scaled = (depthmap_vision_frame / numpy.max(depthmap_vision_frame)) * 255
		shift = ((255 - depth_scaled) / 255 * shift_factor).astype(int)
		
		stereogram = numpy.zeros_like(temp_vision_frame)
		shifted_indices = numpy.maximum(numpy.arange(temp_vision_frame.shape[1])[None, :] + shift[:, :], 0)
		shifted_indices = numpy.clip(shifted_indices, 0, temp_vision_frame.shape[1] - 1)
		stereogram[numpy.arange(temp_vision_frame.shape[0])[:, None], numpy.arange(temp_vision_frame.shape[1])] = temp_vision_frame[numpy.arange(temp_vision_frame.shape[0])[:, None], shifted_indices]
		stereogram = numpy.clip(stereogram, 0, 255)
				
		anaglyph = numpy.zeros_like(temp_vision_frame)
		shift = ((255 - depthmap_vision_frame) / 255 * shift_factor).astype(int)
		x_left = numpy.arange(temp_vision_frame.shape[1])
		x_right = numpy.clip(x_left - shift, 0, temp_vision_frame.shape[1] - 1)
		anaglyph[:, :, 2] = stereogram[:, :, 2]
		anaglyph[:, :, 0] = temp_vision_frame[:, :, 0]
		anaglyph[:, :, 1] = temp_vision_frame[:, :, 1]
     
		depthmap_vision_frame = anaglyph
		
	else:
		depthmap_vision_frame = cv2.cvtColor(depthmap_vision_frame, cv2.COLOR_GRAY2BGR)
	
	
	return depthmap_vision_frame


def forward(depthmap_vision_frame: VisionFrame) -> VisionFrame:
	frame_depthmap = get_inference_pool().get('frame_depthmap')
	model_type = get_model_options().get('type')

	with thread_semaphore():
		depthmap_vision_frame = frame_depthmap.run(None, {'image': depthmap_vision_frame})[0]

		depth_map = depthmap_vision_frame[0, 0]

		min_val = depth_map.min()
		max_val = depth_map.max()
		if max_val > min_val:
			depth_map = (depth_map - min_val) / (max_val - min_val)
		else:
			depth_map = numpy.zeros_like(depth_map)

		depth_map = (depth_map * 255.0).clip(0, 255).astype(numpy.uint8)
		#depth_map = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)

	return depth_map


def prepare_temp_frame(temp_vision_frame : VisionFrame) -> VisionFrame:
	model_type = get_model_options().get('type')
	
#	reszize frame to original depth_anything input size = 518x518 
	temp_vision_frame = cv2.resize(temp_vision_frame,(518,518))
#	or resize to half width and height for faster inference:
#	h, w = temp_vision_frame.shape[:2]
#	temp_vision_frame = cv2.resize(temp_vision_frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
	
	temp_vision_frame = cv2.cvtColor(temp_vision_frame, cv2.COLOR_BGR2RGB)
	temp_vision_frame = temp_vision_frame.astype(numpy.float32) / 255.0
	temp_vision_frame = numpy.transpose(temp_vision_frame, (2, 0, 1))        
	temp_vision_frame = numpy.expand_dims(temp_vision_frame, axis=0).astype(numpy.float32)
 
	return temp_vision_frame

def process_frame(inputs : depthmapInputs) -> VisionFrame:
	vision_frame = inputs.get('temp_vision_frame')
	masked_frame = depthmap_frame(vision_frame)
		
	return masked_frame
		

