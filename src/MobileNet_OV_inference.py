# ---------------------------------------------------------------------
# MobileNet OV inference 
# Copyright (c) 2025 Patrick 
#
# Terms of Use:
# This software is provided strictly for educational and research purposes. 
# Redistribution, modification, or commercial use without explicit permission 
# from the author is prohibited.
#
# Disclaimer:
# This code is for educational use only. If you believe it infringes on
# any copyright or license, please notify the author for correction.
# ---------------------------------------------------------------------

class TextColor:
    RED = "\033[31m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    RESET = "\033[0m"
    

import argparse

parser = argparse.ArgumentParser(description = "help msg")
parser.add_argument("-i", "--input", help = "input image")
parser.add_argument("-m", "--model", help = "input model")

args = parser.parse_args()
if not args.input:
    print (f"{TextColor.RED}\n ERROR no image read")
    exit(1)
else:
    img_location = args.input

if not args.model:
    print (f"{TextColor.RED}\n ERROR no input model read")
    exit(1)
else:
    if "et_v1" in args.model:
        model_path = args.model
    elif "et_v2" in args.model:
        model_path = args.model
    elif "et_v3" in args.model:
        model_path = args.model
    else:
        print(f"{TextColor.RED}\n ERROR no valid model version labled")
        exit(1)


from openvino.inference_engine import IECore
import onnx
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.models import Model
from tensorflow.keras.applications import imagenet_utils
import torch
from torch.utils.data import dataset
import torchvision
from time import process_time

print(f"{TextColor.BLUE}\n\n---------------- INFO ----------------\n\n \
        Loading Model ...")

# load the model
ie_core = IECore()

# read the network and corresponding weights from file
model = ie_core.read_network(model=model_path)

# load the model on the CPU (you can choose manually CPU, GPU, MYRIAD etc.) 
# or let the engine choose best available device (AUTO)
compiled_model = ie_core.load_network(network=model, device_name="CPU")

# get input and output names of modelVersions
input_layer = next(iter(compiled_model.input_info))
output_layer = next(iter(compiled_model.outputs.keys()))
print(f"{TextColor.BLUE} Model {model_path} loaded")


def prepare_image(file):
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    if "et_v1" in model_path:
        return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
    elif "et_v2" in model_path:
        return tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expanded_dims)
    elif "et_v3" in model_path:
        return tf.keras.applications.mobilenet_v3.preprocess_input(img_array_expanded_dims)


print(f"{TextColor.BLUE} Preparing image ...")
# make the prediction
preprocessed_image = prepare_image(img_location)
print(f"{TextColor.BLUE}\n Starting inference ...\n\n")

process_time_start = process_time()
predictions = compiled_model.infer(inputs={input_layer: preprocessed_image})[output_layer]
process_time_stop = process_time()

# print top-5 possible resuls
results = imagenet_utils.decode_predictions(predictions)
print (f"{TextColor.GREEN} Inference results: ")
print(results)
print(f"{TextColor.GREEN}\n\ntime spent: " + str(process_time_stop-process_time_start))
print(f"{TextColor.RESET}\n\n================= END =================\n \
        END of function\n")
