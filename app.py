from typing import Tuple

import gradio as gr
import numpy as np
import supervision as sv
from inference import get_model

MARKDOWN = """
# YOLO-Arena

Powered by Roboflow [Inference](https://github.com/roboflow/inference) and 
[Supervision](https://github.com/roboflow/supervision).
"""

IMAGE_EXAMPLES = [["https://media.roboflow.com/dog.jpeg", 0.3]]

YOLO_V8_MODEL = get_model(model_id="yolov8s-640")
YOLO_NAS_MODEL = get_model(model_id="coco/14")
YOLO_V9_MODEL = get_model(model_id="coco/17")

LABEL_ANNOTATORS = sv.LabelAnnotator(text_color=sv.Color.black())
BOUNDING_BOX_ANNOTATORS = sv.BoundingBoxAnnotator()


def process_image(
    input_image: np.ndarray, confidence_threshold: float, iou_threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    yolo_v8_result = YOLO_V8_MODEL.infer(
        input_image, confidence=confidence_threshold, iou_threshold=iou_threshold
    )[0]
    yolo_v8_detections = sv.Detections.from_inference(yolo_v8_result)

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(
            yolo_v8_detections["class_name"], yolo_v8_detections.confidence
        )
    ]

    yolo_v8_annotated_image = input_image.copy()
    yolo_v8_annotated_image = BOUNDING_BOX_ANNOTATORS.annotate(
        scene=yolo_v8_annotated_image, detections=yolo_v8_detections
    )
    yolo_v8_annotated_image = LABEL_ANNOTATORS.annotate(
        scene=yolo_v8_annotated_image, detections=yolo_v8_detections, labels=labels
    )

    yolo_nas_result = YOLO_NAS_MODEL.infer(
        input_image, confidence=confidence_threshold, iou_threshold=iou_threshold
    )[0]
    yolo_nas_detections = sv.Detections.from_inference(yolo_nas_result)

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(
            yolo_nas_detections["class_name"], yolo_nas_detections.confidence
        )
    ]

    yolo_nas_annotated_image = input_image.copy()
    yolo_nas_annotated_image = BOUNDING_BOX_ANNOTATORS.annotate(
        scene=yolo_nas_annotated_image, detections=yolo_nas_detections
    )
    yolo_nas_annotated_image = LABEL_ANNOTATORS.annotate(
        scene=yolo_nas_annotated_image, detections=yolo_nas_detections, labels=labels
    )

    yolo_v9_result = YOLO_V9_MODEL.infer(
        input_image, confidence=confidence_threshold, iou_threshold=iou_threshold
    )[0]
    yolo_v9_detections = sv.Detections.from_inference(yolo_v9_result)

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(
            yolo_v9_detections["class_name"], yolo_v9_detections.confidence
        )
    ]

    yolo_v9_annotated_image = input_image.copy()
    yolo_v9_annotated_image = BOUNDING_BOX_ANNOTATORS.annotate(
        scene=yolo_v9_annotated_image, detections=yolo_v9_detections
    )
    yolo_v9_annotated_image = LABEL_ANNOTATORS.annotate(
        scene=yolo_v9_annotated_image, detections=yolo_v9_detections, labels=labels
    )

    return yolo_v8_annotated_image, yolo_nas_annotated_image, yolo_v9_annotated_image


confidence_threshold_component = gr.Slider(
    minimum=0,
    maximum=1.0,
    value=0.3,
    step=0.01,
    label="Confidence Threshold",
    info=(
        "The confidence threshold for the YOLO model. Lower the threshold to "
        "reduce false negatives, enhancing the model's sensitivity to detect "
        "sought-after objects. Conversely, increase the threshold to minimize false "
        "positives, preventing the model from identifying objects it shouldn't."
    ),
)

iou_threshold_component = gr.Slider(
    minimum=0,
    maximum=1.0,
    value=0.5,
    step=0.01,
    label="IoU Threshold",
    info=(
        "The Intersection over Union (IoU) threshold for non-maximum suppression. "
        "Decrease the value to lessen the occurrence of overlapping bounding boxes, "
        "making the detection process stricter. On the other hand, increase the value "
        "to allow more overlapping bounding boxes, accommodating a broader range of "
        "detections."
    ),
)


with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Accordion("Configuration", open=False):
        confidence_threshold_component.render()
        iou_threshold_component.render()
    with gr.Row():
        input_image_component = gr.Image(type="numpy", label="Input Image")
        yolo_v8_output_image_component = gr.Image(type="numpy", label="YOLOv8 Output")
    with gr.Row():
        yolo_nas_output_image_component = gr.Image(
            type="numpy", label="YOLO-NAS Output"
        )
        yolo_v9_output_image_component = gr.Image(type="numpy", label="YOLOv9 Output")
    submit_button_component = gr.Button(value="Submit", scale=1, variant="primary")
    gr.Examples(
        fn=process_image,
        examples=IMAGE_EXAMPLES,
        inputs=[
            input_image_component,
            confidence_threshold_component,
            iou_threshold_component,
        ],
        outputs=[
            yolo_v8_output_image_component,
            yolo_nas_output_image_component,
            yolo_v9_output_image_component,
        ],
    )

    submit_button_component.click(
        fn=process_image,
        inputs=[
            input_image_component,
            confidence_threshold_component,
            iou_threshold_component,
        ],
        outputs=[
            yolo_v8_output_image_component,
            yolo_nas_output_image_component,
            yolo_v9_output_image_component,
        ],
    )

demo.launch(debug=False, show_error=True, max_threads=1)
