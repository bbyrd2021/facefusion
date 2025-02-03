from time import sleep
from typing import List

from onnxruntime import InferenceSession

from facefusion import process_manager, state_manager
from facefusion.app_context import detect_app_context
from facefusion.execution import create_inference_execution_providers
from facefusion.thread_helper import thread_lock
from facefusion.typing import DownloadSet, ExecutionProvider, InferencePool, InferencePoolSet

INFERENCE_POOLS : InferencePoolSet =\
{
	'cli': {}, #type:ignore[typeddict-item]
	'ui': {} #type:ignore[typeddict-item]
}


def get_inference_pool(model_context : str, model_sources : DownloadSet) -> InferencePool:
	global INFERENCE_POOLS

	with thread_lock():
		while process_manager.is_checking():
			sleep(0.5)
		app_context = detect_app_context()
		inference_context = get_inference_context(model_context)

		# Log the context information
		print(f"[INFERENCE POOL] App context: {app_context}, Inference context: {inference_context}")

		if app_context == 'cli' and INFERENCE_POOLS.get('ui').get(inference_context):
			INFERENCE_POOLS['cli'][inference_context] = INFERENCE_POOLS.get('ui').get(inference_context)
			print(f"[INFERENCE POOL] Shared pool from 'ui' to 'cli' for context: {inference_context}")
		if app_context == 'ui' and INFERENCE_POOLS.get('cli').get(inference_context):
			INFERENCE_POOLS['ui'][inference_context] = INFERENCE_POOLS.get('cli').get(inference_context)
			print(f"[INFERENCE POOL] Shared pool from 'cli' to 'ui' for context: {inference_context}")
		if not INFERENCE_POOLS.get(app_context).get(inference_context):
			print(f"[INFERENCE POOL] Creating new inference pool for context: {inference_context}")
			INFERENCE_POOLS[app_context][inference_context] = create_inference_pool(model_sources, state_manager.get_item('execution_device_id'), state_manager.get_item('execution_providers'))
		else:
			print(f"[INFERENCE POOL] Reusing existing inference pool for context: {inference_context}")	

		return INFERENCE_POOLS.get(app_context).get(inference_context)


def create_inference_pool(model_sources : DownloadSet, execution_device_id : str, execution_providers : List[ExecutionProvider]) -> InferencePool:
	inference_pool : InferencePool = {}

	for model_name in model_sources.keys():
		model_path = model_sources.get(model_name).get('path')
		print(f"[INFERENCE POOL] Loading model '{model_name}' from path: {model_path}")
		inference_pool[model_name] = create_inference_session(model_sources.get(model_name).get('path'), execution_device_id, execution_providers)
		print(f"[INFERENCE POOL] Model '{model_name}' loaded successfully.")
	return inference_pool


def clear_inference_pool(model_context : str) -> None:
	global INFERENCE_POOLS

	app_context = detect_app_context()
	inference_context = get_inference_context(model_context)

	if INFERENCE_POOLS.get(app_context).get(inference_context):
		print(f"[INFERENCE POOL] Clearing inference pool for context: {inference_context}")
		del INFERENCE_POOLS[app_context][inference_context]


def create_inference_session(model_path : str, execution_device_id : str, execution_providers : List[ExecutionProvider]) -> InferenceSession:
	inference_execution_providers = create_inference_execution_providers(execution_device_id, execution_providers)
	print(f"[MODEL SESSION] Creating InferenceSession for model at {model_path} using providers: {inference_execution_providers}")
	session = InferenceSession(model_path, providers=inference_execution_providers)
	print(f"[MODEL SESSION] InferenceSession created for model at {model_path}")
	return session


def get_inference_context(model_context : str) -> str:
	inference_context = model_context + '.' + '_'.join(state_manager.get_item('execution_providers'))
	return inference_context
