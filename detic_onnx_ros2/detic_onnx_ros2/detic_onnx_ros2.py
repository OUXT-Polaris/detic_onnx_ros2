import os
from typing import Any, List, Dict, Tuple
import requests
import onnxruntime
import PIL.Image

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from detic_onnx_ros2_msg.msg import (
    SegmentationInfo,
    Segmentation,
    Polygon,
    PointOnImage,
)

from cv_bridge import CvBridge
from detic_onnx_ros2.imagenet_21k import IN21K_CATEGORIES
from detic_onnx_ros2.lvis import LVIS_CATEGORIES as LVIS_V1_CATEGORIES
from ament_index_python import get_package_share_directory
from detic_onnx_ros2.color import random_color, color_brightness
import time


class DeticNode(Node):
    def __init__(self):
        super().__init__("detic_node")
        self.declare_parameter("detection_width", 800)
        self.detection_width: int = self.get_parameter("detection_width").value
        self.weight_and_model = self.download_onnx(
            "Detic_C2_SwinB_896_4x_IN-21K+COCO_lvis_op16.onnx"
        )
        self.session = onnxruntime.InferenceSession(
            self.weight_and_model,
            providers=["CPUExecutionProvider"],  # "CUDAExecutionProvider"],
        )
        self.image_publisher = self.create_publisher(Image, "detic_result/image", 10)
        self.segmentation_publisher = self.create_publisher(
            SegmentationInfo, "segmentationinfo", 10
        )
        self.subscription = self.create_subscription(
            Image,
            "image_raw",
            self.image_callback,
            10,
        )
        self.bridge = CvBridge()

    def download_onnx(
        self,
        model: str,
        base_url: str = "https://storage.googleapis.com/ailia-models/detic/",
    ) -> str:
        download_directory = os.path.join(
            get_package_share_directory("detic_onnx_ros2")
        )
        weight_path = os.path.join(download_directory, model)
        if not os.path.exists(weight_path):
            self.get_logger().info("Start downloading model")
            with open(weight_path, mode="wb") as f:
                f.write(requests.get(base_url + model).content)
        else:
            self.get_logger().info(
                "Model was found at path : " + weight_path + " skipping..."
            )
        return weight_path

    def get_lvis_meta_v1(self) -> Dict[str, List[str]]:
        thing_classes = [k["synonyms"][0] for k in LVIS_V1_CATEGORIES]
        meta = {"thing_classes": thing_classes}
        return meta

    # This function comes from https://github.com/axinc-ai/ailia-models/blob/da1c277b602606586cd83943ef6b23eb705ec604/object_detection/detic/dataset_utils.py#L14-L18
    def get_in21k_meta_v1(self) -> Dict[str, List[str]]:
        thing_classes = IN21K_CATEGORIES
        meta = {"thing_classes": thing_classes}
        return meta

    def draw_predictions(
        self, image: np.ndarray, detection_results: Any, vocabulary: str
    ) -> Tuple[np.ndarray, List[Segmentation]]:
        segmentations: List[Segmentation] = []
        width = image.shape[1]
        height = image.shape[0]

        boxes = detection_results["boxes"].astype(np.int64)
        scores = detection_results["scores"]
        classes = detection_results["classes"].tolist()
        masks = detection_results["masks"].astype(np.uint8)

        class_names = (
            self.get_lvis_meta_v1()
            if vocabulary == "lvis"
            else self.get_in21k_meta_v1()
        )["thing_classes"]
        labels = [class_names[i] for i in classes]
        labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]

        num_instances = len(boxes)

        np.random.seed()
        assigned_colors = [random_color(maximum=255) for _ in range(num_instances)]

        areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs]
            labels = [labels[k] for k in sorted_idxs]
            masks = [masks[idx] for idx in sorted_idxs]
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]

        default_font_size = int(max(np.sqrt(height * width) // 90, 10))

        for i in range(num_instances):
            segmentation: Segmentation = Segmentation()
            color = assigned_colors[i]
            color = (int(color[0]), int(color[1]), int(color[2]))
            image_b = image.copy()

            # draw box
            x0, y0, x1, y1 = boxes[i]
            cv2.rectangle(
                image_b,
                (x0, y0),
                (x1, y1),
                color=color,
                thickness=default_font_size // 4,
            )
            segmentation.bounding_box.xmin = int(min(x0, x1))
            segmentation.bounding_box.xmax = int(max(x0, x1))
            segmentation.bounding_box.ymin = int(min(y0, y1))
            segmentation.bounding_box.ymax = int(max(y0, y1))

            # draw segment
            polygons = self.mask_to_polygons(masks[i])
            for points in polygons:
                polygon = Polygon()
                points = np.array(points).reshape((1, -1, 2)).astype(np.int32)
                for i in range(points[0].shape[0]):
                    point_on_image = PointOnImage()
                    point_on_image.x = int(points[0][i][0])
                    point_on_image.y = int(points[0][i][1])
                    polygon.points.append(point_on_image)
                cv2.fillPoly(image_b, pts=[points], color=color)
                segmentation.polygons.append(polygon)
            segmentations.append(segmentation)

            image = cv2.addWeighted(image, 0.5, image_b, 0.5, 0)

        for i in range(num_instances):
            color = assigned_colors[i]
            color_text = color_brightness(color, brightness_factor=0.7)

            color = (int(color[0]), int(color[1]), int(color[2]))
            color_text = (int(color_text[0]), int(color_text[1]), int(color_text[2]))

            x0, y0, x1, y1 = boxes[i]

            SMALL_OBJECT_AREA_THRESH = 1000
            instance_area = (y1 - y0) * (x1 - x0)

            # for small objects, draw text at the side to avoid occlusion
            text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
            if instance_area < SMALL_OBJECT_AREA_THRESH or y1 - y0 < 40:
                if y1 >= height - 5:
                    text_pos = (x1, y0)
                else:
                    text_pos = (x0, y1)

            # draw label
            x, y = text_pos
            text = labels[i]
            font = cv2.FONT_HERSHEY_SIMPLEX
            height_ratio = (y1 - y0) / np.sqrt(height * width)
            font_scale = np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5
            font_thickness = 1
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_w, text_h = text_size
            cv2.rectangle(
                image, text_pos, (int(x + text_w * 0.6), y + text_h), (0, 0, 0), -1
            )
            cv2.putText(
                image,
                text,
                (x, y + text_h - 5),
                fontFace=font,
                fontScale=font_scale * 0.6,
                color=color_text,
                thickness=font_thickness,
                lineType=cv2.LINE_AA,
            )

        return image, segmentations

    def mask_to_polygons(self, mask: np.ndarray) -> List[Any]:
        # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
        # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
        # Internal contours (holes) are placed in hierarchy-2.
        # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
        mask = np.ascontiguousarray(
            mask
        )  # some versions of cv2 does not support incontiguous arr
        res = cv2.findContours(
            mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return []
        res = res[-2]
        res = [x.flatten() for x in res]
        # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
        # We add 0.5 to turn them into real-value coordinate space. A better solution
        # would be to first +0.5 and then dilate the returned polygon by 0.5.
        return [x + 0.5 for x in res if len(x) >= 6]

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        height, width, _ = image.shape
        image = image[:, :, ::-1]  # BGR -> RGB
        size = self.detection_width
        max_size = self.detection_width
        scale = size / min(height, width)
        if height < width:
            oh, ow = size, scale * width
        else:
            oh, ow = scale * height, size
        if max(oh, ow) > max_size:
            scale = max_size / max(oh, ow)
            oh = oh * scale
            ow = ow * scale
        ow = int(ow + 0.5)
        oh = int(oh + 0.5)
        image = np.asarray(
            PIL.Image.fromarray(image).resize((ow, oh), PIL.Image.BILINEAR)
        )
        image = image.transpose((2, 0, 1))  # HWC -> CHW
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        return image

    def image_callback(self, msg):
        input_image = self.bridge.imgmsg_to_cv2(msg)

        vocabulary = "lvis"

        class_names = (
            self.get_lvis_meta_v1()
            if vocabulary == "lvis"
            else self.get_in21k_meta_v1()
        )["thing_classes"]

        image = self.preprocess(image=input_image)
        input_height = image.shape[2]
        input_width = image.shape[3]
        inference_start_time = time.perf_counter()
        boxes, scores, classes, masks = self.session.run(
            None,
            {
                "img": image,
                "im_hw": np.array([input_height, input_width]).astype(np.int64),
            },
        )
        inference_end_time = time.perf_counter()
        self.get_logger().info(
            "Inference takes "
            + str(inference_end_time - inference_start_time)
            + " [sec]"
        )
        draw_mask = masks
        masks = masks.astype(np.uint8)
        draw_classes = classes
        draw_boxes = boxes
        draw_scores = scores

        labels = [class_names[i] for i in classes]
        areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs]
            labels = [labels[k] for k in sorted_idxs]
            masks = [masks[idx] for idx in sorted_idxs]
        scores = scores.astype(np.float32)
        detection_results = {
            "boxes": draw_boxes,
            "scores": draw_scores,
            "classes": draw_classes,
            "masks": draw_mask,
        }
        visualization, segmentations = self.draw_predictions(
            cv2.cvtColor(
                cv2.resize(input_image, (input_width, input_height)), cv2.COLOR_BGR2RGB
            ),
            detection_results,
            "lvis",
        )
        segmentation_info = SegmentationInfo()
        segmentation_info.header = msg.header
        segmentation_info.segmentations = segmentations
        self.segmentation_publisher.publish(segmentation_info)
        self.image_publisher.publish(self.bridge.cv2_to_imgmsg(visualization, "bgr8"))


def main(args=None):
    rclpy.init(args=args)
    detic_node = DeticNode()
    rclpy.spin(detic_node)
    detic_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
