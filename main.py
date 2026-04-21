from ultralytics import YOLO
from PIL import Image, ImageDraw
import gradio as gr

pt_model_path = "yolo11n.pt"
onnx_model_path = "yolo11n.onnx"
image_path = "image-2.png"

origianl_image = Image.open(image_path)
deteced_image_pt = origianl_image.copy()
deteced_image_onnx = origianl_image.copy()

# Load a model
model = YOLO(pt_model_path)  # load an official model

# Predict with the model
pt_results = model(image_path)  # predict on an image

# Access the results
for result in pt_results:
    xywh = result.boxes.xywh  # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf  # confidence score of each box
    print("--------------------------------")
    print(result)
    print("--------------------------------")
    for box, name, conf in zip(xyxy, names, confs):
        x1, y1, x2, y2 = box
        draw = ImageDraw.Draw(deteced_image_pt)
        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
        draw.text((x1, y1), name + " " + str(conf), fill="blue")

deteced_image_pt.save("deteced_image_pt.png")
# deteced_image_pt.show()


def onnx_convert(model, input_file, output_file):

    model = model.export(format="onnx")
    # model.save(output_file)

onnx_convert(model, pt_model_path, onnx_model_path)

onnx_model = YOLO(onnx_model_path)
onnx_results = onnx_model(image_path)

for result in onnx_results:
    xywh = result.boxes.xywh  # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf  # confidence score of each box
    print("--------------------------------")
    print(result)
    print("--------------------------------")
    for box, name, conf in zip(xyxy, names, confs):
        x1, y1, x2, y2 = box
        draw = ImageDraw.Draw(deteced_image_onnx)
        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
        draw.text((x1, y1), name + " " + str(conf), fill="blue")

deteced_image_onnx.save("deteced_image_onnx.png")
# deteced_image_onnx.show()

