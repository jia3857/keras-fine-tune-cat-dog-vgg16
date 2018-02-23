import tensorflow
import sys

from tensorflow.python.client import device_lib

def get_available_gpus():
  local_device_protos = device_lib.list_local_devices()
  print ([x.name for x in local_device_protos if x.device_type == 'GPU'])
  return [x.name for x in local_device_protos if x.device_type == 'GPU']

available_gpus =  get_available_gpus()

assert(len(available_gpus) > 0)
