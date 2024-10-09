import os
import argparse

from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import time

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

from vitpose_model import ViTPoseModel

import json
from typing import Dict, Optional

from server_tool.shared_memory import SharedMemoryManager
from server_tool.sync_unit import SyncUnit
import server_tool.socket_util as socket_util


def process(img, detector, cpm):
    img_cv2 = img[:, :, ::-1].copy()
    # resize
    # ratio = 2
    # img_cv2 = cv2.resize(img_cv2, None, fx=1 / ratio, fy=1 / ratio)

    det_out = detector(img_cv2)

    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()

    # Detect human keypoints for each person
    vitposes_out = cpm.predict_pose(
        img,
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
    )

    bboxes = []
    is_right = []

    # Use hands based on hand keypoint detections
    for vitposes in vitposes_out:
        left_hand_keyp = vitposes['keypoints'][-42:-21]
        right_hand_keyp = vitposes['keypoints'][-21:]

        # Rejecting not confident detections
        keyp = left_hand_keyp
        valid = keyp[:, 2] > 0.5
        if sum(valid) > 3:
            bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
            bboxes.append(bbox)
            is_right.append(0)
        keyp = right_hand_keyp
        valid = keyp[:, 2] > 0.5
        if sum(valid) > 3:
            bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
            bboxes.append(bbox)
            is_right.append(1)

    if len(bboxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.ones((0,), dtype=np.int32)

    boxes = np.stack(bboxes).astype(np.float32)
    # boxes = boxes * ratio
    right = np.stack(is_right).astype(np.int32)
    return boxes, right


def pair(bboxes_list, right_list):
    bboxes_list_left, bboxes_list_right = [], []
    for bboxes, rightv in zip(bboxes_list, right_list):
        # pick largest bbox for left hand and right hand
        bbox_left_sel, bbox_right_sel = None, None
        bbox_left_area, bbox_right_area = None, None
        for bbox, right in zip(bboxes, rightv):
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if right == 0:
                if bbox_left_area is None or bbox_left_area < area:
                    bbox_left_area = area
                    bbox_left_sel = bbox
            else:
                if bbox_right_area is None or bbox_right_area < area:
                    bbox_right_area = area
                    bbox_right_sel = bbox
        bboxes_list_left.append(bbox_left_sel)
        bboxes_list_right.append(bbox_right_sel)
    res = {
        "left": bboxes_list_left,
        "right": bboxes_list_right,
    }
    return res


def main(camera_info, video_shape, host, port):
    camera_name_list = list(camera_info.values())

    # load param
    cam_extr_map, cam_intr_map = {}, {}
    for cam_name in camera_name_list:
        _recv = SharedMemoryManager(
            name=f"cam_extr__{cam_name}",
            type=1,
            shape=(4, 4),
            dtype=np.float32,
            timeout=60,
        )
        _recv.unregister()
        cam_extr, _ = _recv.execute()
        _recv.close()
        cam_extr_map[cam_name] = cam_extr

        _recv = SharedMemoryManager(
            name=f"cam_intr__{cam_name}",
            type=1,
            shape=(3, 3),
            dtype=np.float32,
            timeout=60,
        )
        _recv.unregister()
        cam_intr, _ = _recv.execute()
        _recv.close()
        cam_intr_map[cam_name] = cam_intr

    # load (synced) image
    recv_list = []
    for cam_name in camera_name_list:
        _recv = SharedMemoryManager(
            name=f"sync__{cam_name}",
            type=1,
            shape=(video_shape[1], video_shape[0], 3),
            dtype=np.uint8,
            timeout=60,
        )
        _recv.unregister()
        recv_list.append(_recv)
    sync_unit = SyncUnit(recv_list)

    # load modeel
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hamer
    cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    s = socket_util.bind(host, port)
    while True:
        conn, addr = s.accept()
        with conn:
            inp = socket_util.conn_recv(conn)
            if inp is None:
                break
            # process
            import time
            img_list, ts = sync_unit.execute()
            bbox_list, right_list = [], []
            for img in img_list:
                bbox, right = process(img, detector, cpm)
                bbox_list.append(bbox)
                right_list.append(right)
            # pair result
            payload = pair(bbox_list, right_list)
            payload["timestamp"] = ts
            try:
                socket_util.conn_resp(conn, payload)
            except BrokenPipeError:
                pass
    s.close()

    for _recv in recv_list:
        _recv.close()

    print("hamer server end")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_info", type=str, required=True)
    parser.add_argument("--video_shape", type=str, required=True)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=11311)
    args = parser.parse_args()
    # parse camera_info
    camera_info_str = args.camera_info
    camera_info = dict(el.split(":", 2) for el in camera_info_str.split(","))
    # parse video_shape
    video_shape_str = args.video_shape
    _split = video_shape_str.split("x", 2)
    video_shape = (int(_split[0]), int(_split[1]))
    # server
    main(camera_info, video_shape, args.host, args.port)
