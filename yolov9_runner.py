import cv2
import time
import yaml
import numpy as np
from typing import Tuple, List
from rknnlite.api import RKNNLite


class_list = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", 
              "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", 
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
              "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", 
                "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", 
                "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
class YOLOv9:
    def __init__(self,
                 model_path: str,
                 class_list: list,
                 score_threshold: float = 0.25,
                 conf_thresold: float = 0.25,
                 iou_threshold: float = 0.4,
                 width: int = 640,
                 height: int = 640,
                 plat_form: str = "rk3588"):
        self.model_path = model_path
        self.class_list = class_list

        self.score_threshold = score_threshold
        self.conf_thresold = conf_thresold
        self.iou_threshold = iou_threshold
        self.image_width, self.image_height = width, height
        self.original_size = (0, 0)
        self.plat_form = plat_form
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            raise Exception("Failed to load model")
        if self.plat_form == "rk3588":
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        else:
            ret = self.rknn.init_runtime()
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.original_size = image_rgb.shape[:2]
        resized = cv2.resize(image_rgb, (self.image_width, self.image_height))
        resized = np.expand_dims(resized, axis=0)
        
        return resized
    
    def xywh2xyxy(self, x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y 
    
    def postprocess(self, outputs):
        predictions = np.squeeze(outputs).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_thresold, :]
        scores = scores[scores > self.conf_thresold]
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Rescale box
        boxes = predictions[:, :4]
        
        input_shape = np.array([self.image_width, self.image_height, self.image_width, self.image_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.original_size[1], self.original_size[0], self.original_size[1], self.original_size[0]])
        boxes = boxes.astype(np.int32)
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=self.score_threshold, nms_threshold=self.iou_threshold)
        detections = []
        for bbox, score, label in zip(self.xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):
            detections.append({
                "class_index": label,
                "confidence": score,
                "box": bbox,
                "class_name": self.get_label_name(label)
            })
        return detections
    
    def get_label_name(self, class_id: int) -> str:
        return self.class_list[class_id]
        
    def detect(self, img: np.ndarray) -> List:
        input_tensor = self.preprocess(img)
        outputs = self.rknn.inference([input_tensor],data_format=['nhwc'])[0]
        print(outputs.shape)
        return self.postprocess(outputs)
    
    def draw_detections(self, img, detections: List):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            detections: List of detection result which consists box, score, and class_ids
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        for detection in detections:
            # Extract the coordinates of the bounding box
            x1, y1, x2, y2 = detection['box'].astype(int)
            class_id = detection['class_index']
            confidence = detection['confidence']

            # Retrieve the color for the class ID
            color = (0, 255, 0)

            # Draw the bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Create the label text with class name and score
            label = f"{self.class_list[class_id]}: {confidence:.2f}"

            # Calculate the dimensions of the label text
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Calculate the position of the label text
            label_x = x1
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

            # Draw a filled rectangle as the background for the label text
            cv2.rectangle(
                img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
            )

            # Draw the label text on the image
            cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

if __name__=="__main__":

    weight_path = "rknn_model/yolov9c.rknn"
    image = cv2.imread("bus.jpg")
    detector = YOLOv9(model_path=weight_path,
                      class_list=class_list,
                    )
                      
    detections = detector.detect(image)
    print(detections)
    detector.draw_detections(image, detections=detections)
    
    cv2.imshow("Tambang Preview", image)
    cv2.imwrite("output.jpg", image)
    cv2.waitKey(3000)