import cv2
import json
import os
import re
from PIL import Image
from tqdm import tqdm

import torch
from collections import Counter
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg

# model file
config_file = "your_folder_path/YOLO-World-master/configs/pretrain/configs_pretrain_yolo_world_xl_t2i_bn_2e-4_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
checkpoint = "your_folder_path/YOLO-World-master/demo/pretrained_weights/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain.pth"

cfg = Config.fromfile(config_file)
cfg.work_dir = os.path.join('./work_dirs')

# initiate model
cfg.load_from = checkpoint
model = init_detector(cfg, checkpoint=checkpoint, device='cuda')
test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
test_pipeline = Compose(test_pipeline_cfg)

# calculate area of the bbox
def calculate_area(bbox):
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    return width * height

# get 5 points
def calculate_points(xyxy):
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    center_point = (cx, cy)
    upper_third_point = (cx, y1 + 2*(cy - y1) / 3)
    lower_third_point = (cx, y2 - 2*(y2 - cy) / 3)
    left_third_point = (x1 + 2*(cx - x1) / 3, cy)
    right_third_point = (x2 - 2*(x2 - cx) / 3, cy)

    return [center_point, upper_third_point, lower_third_point, left_third_point, right_third_point]

# enframe the video
def enframe_video(video_path, output_frame_base, frame_per_second=1):
    
    if not os.path.exists(output_frame_base):
        os.makedirs(output_frame_base, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        current_second = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        if frame_count % int(fps) == 0:
            frame_filename = os.path.join(output_frame_base, f"{current_second}.jpg")
            cv2.imwrite(frame_filename, frame)
    cap.release()

# sort the frames
def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else 0

# YOLO-World detection
def inference(model, image, texts, test_pipeline, score_thr=0.3, max_dets=100):
    image = cv2.imread(image)
    image = image[:, :, [2, 1, 0]]
    data_info = dict(img=image, img_id=0, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])
    with torch.no_grad():
        output = model.test_step(data_batch)[0]
    pred_instances = output.pred_instances
    # score thresholding
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
    # max detections
    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()
    boxes = pred_instances['bboxes']
    labels = pred_instances['labels']
    scores = pred_instances['scores']
    label_texts = [texts[x][0] for x in labels]
    return boxes, labels, label_texts, scores

# yolo detect a video for 4 times
def yolo_generate(video_path, frame_folder, objects, threshold=0.5):

    # enframe the video
    enframe_video(video_path, frame_folder)

    # sort the frames
    image_files = sorted(
        [f for f in os.listdir(frame_folder) if os.path.isfile(os.path.join(frame_folder, f))],
        key=numerical_sort
    )

    # get the area threshold
    with Image.open(os.path.join(frame_folder, image_files[0])) as img:
        width, height = img.size
    area_threshold = width * height / 100

    frame_count = len(image_files)    # frame number
    detection_interval = max(1, (frame_count // 4)+1)

    all_results = []
    for idx in tqdm(range(0, frame_count, detection_interval), desc="Processing images"):       #disable=True
        image_file = image_files[idx]
        image_path = os.path.join(frame_folder, image_file)

        print(f"starting to detect: {image_path}")
        results = inference(model, image_path, objects, test_pipeline)

        # filter
        label_texts = results[2]
        label_counts = Counter(label_texts)
        duplicate_labels = {label for label, count in label_counts.items() if count > 1}
        mask = [label not in duplicate_labels for label in label_texts]
        filtered_results = (
            results[0][mask],
            results[1][mask],
            [label for i, label in enumerate(label_texts) if mask[i]],
            results[3][mask]
        )

        image_results = []
        for idxx, (box, _, lbl_text, score) in enumerate(zip(*filtered_results)):
            bbox_int = box.astype(int)
            area = calculate_area(bbox_int)
            if score > threshold and area > area_threshold:
                selected_points = calculate_points(bbox_int)
                result = {
                    "序号": idxx + 1,
                    '置信度': score,
                    "类别": lbl_text,
                    "边界框": bbox_int,
                    'random_points':selected_points
                }
                image_results.append(result)

        if image_results:
            all_results.append({"时间/s": idx, "检测结果": image_results})

    return all_results

if __name__ == '__main__':
    p = ''
    result = yolo_generate(p)
    print(result)
