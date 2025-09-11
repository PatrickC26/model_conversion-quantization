# ---------------------------------------------------------------------
# MobileNet model Conversion and Quantization
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

import os
import numpy as np

ROOT = "./" # Locate a location for the main folder

modelVersion = "v1"
modelVersion = "v2"
modelVersion = "v3_Large"
modelVersion = "v3_Small"

# [preprocess] Make the image to required pattern
from keras.preprocessing import image
def prepare_image(file):
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    if "v1" in modelVersion:
        return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
    elif "v2" in modelVersion:
        return tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expanded_dims)
    elif "v3" in modelVersion:
        return tf.keras.applications.mobilenet_v3.preprocess_input(img_array_expanded_dims)

# create a dir if it is empty
def createDir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


# list out the photo in the folder
photo_ls = os.listdir(ROOT)
photo_ext = ('.png', '.jpeg', '.jpg')
photo_ls = [f for f in photo_ls if f.lower().endswith(photo_ext)]


import tensorflow as tf



# load model
import keras
if "v1" in modelVersion:
    model = keras.applications.mobilenet.MobileNet()
elif "v2" in modelVersion:
    model = keras.applications.mobilenet_v2.MobileNetV2()
elif "v3_Large" in modelVersion:
    from tensorflow.keras.applications import MobileNetV3Large
    model = MobileNetV3Large()
elif "v3_Small" in modelVersion:
    from tensorflow.keras.applications import MobileNetV3Small
    model = MobileNetV3Small()
print(f"{TextColor.BLUE}\n\n---------------- INFO ----------------\n\n \
    MobileNet_" + modelVersion + " model loaded")


# make a testing prediction from the origional model
print(" Making an inference from the origional model...")

#load libs 
from tensorflow.keras.applications import imagenet_utils
from time import process_time

# check if photo is in the ROOT folder
preprocessed_image = prepare_image(ROOT + photo_ls[0])
if len(photo_ls) == 0:
    print(f"{TextColor.RED}\n\n---------------- ERROR ----------------\n\n \
            ERROR no image in the folder.\n Please insert one and restart the program \
            \n\n================= END =================\n")
    exit(1)

# use process time to cal. the time take for inference
process_time_start = process_time()
predictions = model.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
process_time_stop = process_time()
print(f"{TextColor.GREEN}\t\tRESULT\ntime spent: \
        {(process_time_stop-process_time_start)}\n {results} \n\n")

# save model for quantize usage
model.save_weights(ROOT + 'keras/mobilenet_' + modelVersion + '.weights.h5')

# convert the model to onnx
import tf2onnx
import onnx
print(f"{TextColor.BLUE}\n\n Starting to convert and quantize \
        \n converting tensorflow to onnx ...")
input_signature = [tf.TensorSpec([None, 224, 224, 3], tf.float32, name='mobilenet_' + modelVersion)]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
onnx.save(onnx_model, ROOT + "onnx/mobilenet_" + modelVersion + ".onnx")

# convert the model to openvino model
import openvino as ov
print(f"{TextColor.GREEN} Saving onnx ...")
ov_model = ov.convert_model(ROOT + "onnx/mobilenet_" + modelVersion + ".onnx")
ov_model.reshape([1,224,224,3])
print(f"{TextColor.GREEN} Saving ov IR fp 32...")
ov.save_model(ov_model, ROOT + "IR/FP32/mobilenet_" + modelVersion + ".xml", compress_to_fp16 = False)
print(f"{TextColor.GREEN} Saving ov IR fp16 ...")
ov.save_model(ov_model, ROOT + "IR/FP16/mobilenet_" + modelVersion + ".xml", compress_to_fp16 = True)




import torch
import nncf
from torchvision import transforms, utils, datasets


# prep for quantization 
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
val_dataset = datasets.ImageFolder(
    root = ROOT + "validation",
    transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
)



# load the calibration data
def transform_fn_ori(data_item):
    images, _ = data_item
    return images.numpy().transpose(0, 2, 3, 1)

val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
calibration_dataset = nncf.Dataset(val_data_loader, transform_fn_ori)


print(f"{TextColor.GREEN} Quantizing with nncf tool ...")
quantized_model = nncf.quantize(ov_model, calibration_dataset)
print(f"{TextColor.GREEN} Saving ov IR int8 ...")
ov.save_model(quantized_model, ROOT + "IR/INT8/mobilenet_" + modelVersion + ".xml")

print (f"{TextColor.GREEN}Model all saved !!")
print(f"{TextColor.RESET}\n \n================= END =================\n END of function\n")