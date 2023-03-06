import os
import time
from collections import namedtuple
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML, FileLink, display
from model.u2net import U2NET, U2NETP
from openvino.runtime import Core

import subprocess
from PIL import Image
import io
import chardet

this_dir, this_filename = os.path.split(__file__) 


IMAGE_DIR = os.path.join(this_dir,"../data/image/")  
model_config = namedtuple("ModelConfig", ["name", "url", "model", "model_args"])

u2net_lite = model_config(
    name="u2net_lite",
    url="https://drive.google.com/uc?id=1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy",
    model=U2NETP,
    model_args=(),
)
u2net = model_config(
    name="u2net",
    url="https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
    model=U2NET,
    model_args=(3, 1),
)
u2net_human_seg = model_config(
    name="u2net_human_seg",
    url="https://drive.google.com/uc?id=1-Yg0cxgrNhHP-016FPdp902BR-kSsA4P",
    model=U2NET,
    model_args=(3, 1),
)

# Set u2net_model to one of the three configurations listed above.

u2net_model = u2net_lite #TODO CHANGE THIS FOR DIFFRENET USAGE  

# The filenames of the downloaded and converted models.
MODEL_DIR = os.path.join(this_dir, "model")
model_path = Path(MODEL_DIR) / u2net_model.name / Path(u2net_model.name).with_suffix(".pth")
onnx_path = model_path.with_suffix(".onnx")
ir_path = model_path.with_suffix(".xml")


if not model_path.exists():
    import gdown

    os.makedirs(name=model_path.parent, exist_ok=True)
    print("Start downloading model weights file... ")
    with open(model_path, "wb") as model_file:
        gdown.download(url=u2net_model.url, output=model_file)
        print(f"Model weights have been downloaded to {model_path}")


# Load the model.
net = u2net_model.model(*u2net_model.model_args)
net.eval()

# Load the weights.
print(f"Loading model weights from: '{model_path}'")
net.load_state_dict(state_dict=torch.load(model_path, map_location="cpu"))

# Save the model if it does not exist yet.
if not model_path.exists():
    print("\nSaving the model")
    torch.save(obj=net.state_dict(), f=str(model_path))
    print(f"Model saved at {model_path}")

if not onnx_path.exists():
    dummy_input = torch.randn(1, 3, 512, 512)
    torch.onnx.export(model=net, args=dummy_input, f=onnx_path, opset_version=11)
    print(f"ONNX model exported to {onnx_path}.")
else:
    print(f"ONNX model {onnx_path} already exists.")


# Construct the command for Model Optimizer.
# Set log_level to CRITICAL to suppress warnings that can be ignored for this demo.
mo_command = f"""mo
                 --input_model "{onnx_path}"
                 --input_shape "[1,3, 512, 512]"
                 --mean_values="[123.675, 116.28 , 103.53]"
                 --scale_values="[58.395, 57.12 , 57.375]"
                 --compress_to_fp16
                 --output_dir "{model_path.parent}"
                 --log_level "CRITICAL"
                 """
mo_command = " ".join(mo_command.split())
print("Model Optimizer command to convert the ONNX model to OpenVINO:")
print(mo_command)



if not ir_path.exists():
    print("Exporting ONNX model to IR... This may take a few minutes.")
    #mo_result = %sx $mo_command #TODO : ca marche sur des notebooks 
    mo_result = subprocess.run(mo_command, shell=True, capture_output=True)
    print("\n".join(mo_result))
else:
    print(f"IR model {ir_path} already exists.")


def read_input_image(file_input) :
    image = Image.open(file_input)
    image = cv2.cvtColor(np.array(image), code=cv2.COLOR_BGR2RGB)

    resized_image = cv2.resize(src=image, dsize=(512, 512))
    # Convert the image shape to a shape and a data type expected by the network
    # for OpenVINO IR model: (1, 3, 512, 512).
    input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)

    return input_image, image 


def predict(image):

    input_image,image = read_input_image(image) 
    # Load the network to OpenVINO Runtime.
    ie = Core()
    model_ir = ie.read_model(model=ir_path)
    compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU")
    # Get the names of input and output layers.
    input_layer_ir = compiled_model_ir.input(0)
    output_layer_ir = compiled_model_ir.output(0)

    # Do inference on the input image.
    start_time = time.perf_counter()
    result = compiled_model_ir([input_image])[output_layer_ir]
    end_time = time.perf_counter()
    print(
        f"Inference finished. Inference time: {end_time-start_time:.3f} seconds, "
        f"FPS: {1/(end_time-start_time):.2f}."
    )


    # Resize the network result to the image shape and round the values
    # to 0 (background) and 1 (foreground).
    # The network result has (1,1,512,512) shape. The `np.squeeze` function converts this to (512, 512).
    resized_result = np.rint(
        cv2.resize(src=np.squeeze(result), dsize=(image.shape[1], image.shape[0]))
    ).astype(np.uint8)

    # Create a copy of the image and set all background values to 255 (white).
    bg_removed_result = image.copy()
    bg_removed_result[resized_result == 0] = 255

    # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))
    # ax[0].imshow(image)
    # ax[1].imshow(resized_result, cmap="gray")
    # ax[2].imshow(bg_removed_result)
    # for a in ax:
    #     a.axis("off")
    
    print('HEY')
    
    image_bytes = image.tobytes() 
    image_bytes_detected_encoding = chardet.detect(image_bytes)["encoding"]

    resized_result_bytes = resized_result.tobytes()
    resized_result_bytes_detected_encoding = chardet.detect(resized_result_bytes)["encoding"]

    result_bytes = resized_result.tobytes() 
    result_bytes_detected_encoding = chardet.detect(result_bytes)["encoding"]


    result = {
        'mask' : resized_result_bytes.decode(resized_result_bytes_detected_encoding),
        'result': result_bytes.decode(result_bytes_detected_encoding)
    }

    return result 

# IMAGE_PATH = Path(IMAGE_DIR) / "coco_hollywood.jpg"
# res = predict(IMAGE_PATH)
# print(type(res['original']))