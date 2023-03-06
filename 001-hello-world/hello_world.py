import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core
from PIL import Image 
import os 

this_dir, this_filename = os.path.split(__file__) 

ie = Core()
#model_path = os.path.join(os.getcwd(),"model/v3-small_224_1.0_float.xml")
#model_path = "../notebooks/001-hello-world/model/v3-small_224_1.0_float.xml"
model_path = os.path.join(this_dir, "model/v3-small_224_1.0_float.xml") 

model = ie.read_model(model=model_path)
compiled_model = ie.compile_model(model=model, device_name="CPU")

output_layer = compiled_model.output(0)

def read_input_image(file_input):
    image = Image.open(file_input)
    image = cv2.cvtColor(np.array(image), code=cv2.COLOR_BGR2RGB)
    return image 

# The MobileNet model expects images in RGB format.
#image = cv2.cvtColor(cv2.imread(filename="../data/image/coco.jpg"), code=cv2.COLOR_BGR2RGB)


def predict(image):

    image = read_input_image(image)
    # Resize to MobileNet image shape.
    input_image = cv2.resize(src=image, dsize=(224, 224))

    # Reshape to model input shape.
    input_image = np.expand_dims(input_image, 0)

    result_infer = compiled_model([input_image])[output_layer]
    result_index = np.argmax(result_infer)

    # Convert the inference result to a class name.
    imagenet_classes = open(os.path.join(this_dir, "../data/datasets/imagenet/imagenet_2012.txt")).read().splitlines()

    # The model description states that for this model, class 0 is a background.
    # Therefore, a background must be added at the beginning of imagenet_classes.
    imagenet_classes = ['background'] + imagenet_classes

    imagenet_classes[result_index]

    result = {
        "classname": imagenet_classes[result_index],
        "precision": np.max(result_infer),
        
    }

    return result 
