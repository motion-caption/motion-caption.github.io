import numpy as np
import json
import os
import math
from PIL import Image
from collections import defaultdict
import torch
from sam2.build_sam import build_sam2_video_predictor


# init
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
sam2_checkpoint = "your_folder_path/segment-anything-2-main/checkpoints/sam2_hiera_large.pt" # base_plus  /  large  /  small  /  tiny
model_cfg = "sam2_hiera_l.yaml"  # b+  /  l  /  s  /  t
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

# get bbox center
def calculate_center(bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y

# get mask centroid
def calculate_centroid(mask):
    rows, cols = np.where(mask)
    if len(rows) == 0 or len(cols) == 0:
        return None
    y = np.mean(rows)
    x = np.mean(cols)
    return int(x), int(y)

# add a click
def add_click_and_get_segmentation(predictor, inference_state, ann_frame_idx, ann_obj_id, points, labels):
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
    return out_obj_ids, out_mask_logits

# check frame similarity
def is_similar(frame1, frame2, center_threshold=30):
    center1 = frame1
    center2 = frame2

    center_diff = math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    if center_diff > center_threshold:
        return False
    
    return True

# merge segments
def merge_centroids(segments):
    used = [False] * 50
    to_delete = set()

    for out_obj_id, frames in segments.items():
        if not used[out_obj_id-1]:
            first_frame_idex = -1 
            used_in = [False] * 50
            for frame_idx, detections in frames.items():
                if first_frame_idex == -1:
                    first_frame_idex = frame_idx
                centroid1, _, category1, _ = detections

                if centroid1 is not None:
                    for out_obj_id2, frames2 in segments.items():
                        centroid2, _, category2, _ = frames2[frame_idx]
                        if centroid2 is not None and category1 == category2 and out_obj_id2 > out_obj_id and not used_in[out_obj_id2-1]:
                            if is_similar(centroid1, centroid2):
                                used[out_obj_id2 - 1] = True
                                for k in range(first_frame_idex,first_frame_idex + len(frames)):
                                    if segments[out_obj_id][k][0] is not None :
                                        segments[out_obj_id][k] = frames[k]
                                    elif segments[out_obj_id2][k][0] is not None :
                                        segments[out_obj_id][k] = frames2[k]
                                to_delete.add(out_obj_id2)
                                used_in[out_obj_id2-1] = True

    # delete
    for obj_id in to_delete:
        if obj_id in segments:
            del segments[obj_id]
        else:
            print(f"Warning: {obj_id} not found in segments")

    return segments

# get segments
def extract_motion_segments(object_trajectory):
    frames = object_trajectory
    trajectory_segments = defaultdict(list)
    current_segment = []
    last_valid_frame = -1
    last_area = -1
    area_decrease_threshold = 1

    for frame_idx, (center, area, category, boxx) in frames.items():
        if center is not None:
            if last_area != 0:
                min_area = min(area, last_area)
                max_area = max(area, last_area)

            if (last_valid_frame != -1 and frame_idx - last_valid_frame > 1):
                if len(current_segment) > 1:
                    trajectory_segments[len(trajectory_segments)].extend(current_segment)
                current_segment = []

            elif min_area < max_area * (1 - area_decrease_threshold):
                if len(current_segment) > 1:
                    trajectory_segments[len(trajectory_segments)].extend(current_segment)
                current_segment = []

            current_segment.append((frame_idx, center, area, category, boxx))
            last_valid_frame = frame_idx
            last_area = area

        else:
            if len(current_segment) > 1:
                trajectory_segments[len(trajectory_segments)].extend(current_segment)
            current_segment = []

    if len(current_segment) > 1:
        trajectory_segments[len(trajectory_segments)].extend(current_segment)

    return trajectory_segments

# get movement from box2
def get_direction(start_position, end_position, width, height, box1, box2):
    start_x, start_y = start_position
    end_x, end_y = end_position
    x_c1, y_c1 = calculate_center(box1)
    x_c2, y_c2 = calculate_center(box2)
    x_threshold = width * 0.45
    y_threshold = height * 0.45
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    delta_x1 = x_c2 - x_c1
    delta_y1 = y_c2 - y_c1
    x_direction = ""
    y_direction = ""
    
    if abs(delta_x) < x_threshold and abs(delta_x1) < x_threshold:
        x_direction = "center"
    elif delta_x > 0:
        x_direction = "right"
    else:
        x_direction = "left"
    
    if abs(delta_y) < y_threshold and abs(delta_y1) < y_threshold:
        y_direction = "center"
    elif delta_y > 0:
        y_direction = "down"
    else:
        y_direction = "up"
    
    if x_direction == "center" and y_direction == "center":
        return "no movement"
    elif x_direction == "center":
        return y_direction
    elif y_direction == "center":
        return x_direction
    else:
        return f"{y_direction}-{x_direction}"

# get movement from segments
def determine_directions(motion_segments, width, height, threshold_dis, object_id):
    results = {}
    for segment_id, segments in motion_segments.items():
        frame_idx1, center1, area1, category, box1 = segments[0]
        frame_idx2, center2, area2,_ , box2= segments[len(segments)-1]
        x1,y1 = center1
        x2,y2 = center2

        threshold1 = width/8
        threshold2 = height/8

        border_proximity_start = (x1 < threshold1 or x1 > width - threshold1 or
                    y1 < threshold2 or y1 > height - threshold2)
        border_proximity_end = (x2 < threshold1 or x2 > width - threshold1 or
                    y2 < threshold2 or y2 > height - threshold2)

        near_border = border_proximity_start or border_proximity_end
        def calculate_bbox_area(bbox):
            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min
            area = width * height
            return area

        box_area1 =calculate_bbox_area(box1)
        box_area2 = calculate_bbox_area(box2)

        if not near_border:
            if area1/area2 >= 4 and box_area1/box_area2 >= 4:
                calculate_bbox_area
                distance = 'away'
            elif area1/area2 <= 0.25 and box_area1/box_area2 <= 0.25:
                distance = 'close'
            else :
                distance = 'no'
        else:
            distance = 'no'

        dis = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

        if dis < threshold_dis:
            direct = 'no movement'
        else:
            direct = get_direction(center1, center2, width, height, box1, box2)

        results[segment_id] = {'motion':direct, 'distance':distance, 'frame_start':frame_idx1, 'frame_end':frame_idx2, 'category':category, 'object_id':object_id}

    return results

# get yolo results, do SAM2
def sam_generate_motion(video_path):
    yolo_base_folder = ''
    folder_path, filename = os.path.split(video_path)
    folder_name = ''
    file_name_without_extension = os.path.splitext(filename)[0]
    yolo_path = ''

    if not os.path.exists(yolo_path):
        print("no yolo")
        return 'no'

    # 获取yolo
    with open(yolo_path, 'r', encoding='utf-8') as json_file:
        all_results = json.load(json_file)

    # 将all_results列表转化为objects字典
    objects = {}
    object_id = 1
    for frame in all_results:
        time = frame["时间/s"]
        for detection in frame["检测结果"]:
            objects[object_id] = []
            objects[object_id].append({"时间/s": time, "序号":detection["序号"], "类别": detection["类别"], "边界框": detection["边界框"], 'random_points': detection["random_points"]})
            object_id += 1

    # 进行追踪
    return sam_yolo(objects, folder_name, file_name_without_extension)


# SAM2 details
def sam_yolo(objects, folder_name, name, predictor=predictor):
    video_frame_dir = ""
    frame_names = [ p for p in os.listdir(video_frame_dir) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    first_image_path = os.path.join(video_frame_dir, frame_names[0])
    with Image.open(first_image_path) as img:
        width, height = img.size
    threshold = np.sqrt(width ** 2 + height ** 2) / 4.5
    
    # init
    inference_state = predictor.init_state(video_path=video_frame_dir)
    predictor.reset_state(inference_state)

    # add clicks
    category_dict = {}
    for object_id, object in objects.items():
        for detection in object:
            center_x, center_y = calculate_center(detection['边界框'])
            points = detection['random_points']
            points = [[x, y] for x, y in points]
            points = np.array(points, dtype=np.float32)
            labels = np.array([1, 1, 1, 1, 1], np.int32)
            out_obj_ids, out_mask_logits = add_click_and_get_segmentation(predictor, inference_state, detection['时间/s'], object_id, points, labels)
            category_dict[object_id] = [detection['类别'], detection['时间/s'], center_x, center_y]

    # propagate in video
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        for i, out_obj_id in enumerate(out_obj_ids):
            # get mask
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()
            area = np.sum(mask)
            centroid = calculate_centroid(mask[0])
            category = category_dict[out_obj_id][0]
            y_indices, x_indices = np.where(mask[0])
            if x_indices.size > 0 and y_indices.size > 0:
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
            else:
                x_min, x_max = 0, 0
                y_min, y_max = 0, 0
            bbox_coordinates = (x_min, y_min, x_max, y_max)
            
            if out_obj_id not in video_segments:
                video_segments[out_obj_id] = {}
            video_segments[out_obj_id][out_frame_idx] = (centroid, area, category, bbox_coordinates)

    # delete lonely frme
    to_delete = set()
    for out_obj_id,frames in video_segments.items():
        frame_keys = sorted(frames.keys())
        for i in range(len(frame_keys)):
            curr_frame_idx = frame_keys[i]
            curr_area = frames[curr_frame_idx][1]
            if i > 0:
                prev_detection = frames[frame_keys[i - 1]][0]
                prev_area = frames[frame_keys[i - 1]][1]
                category = frames[frame_keys[i - 1]][2]
            else:
                prev_detection = None
                prev_area = 0
            if i < len(frame_keys) - 1:
                next_detection = frames[frame_keys[i + 1]][0]
            else:
                next_detection = None

            if curr_area!=0 and prev_area!=0:
                if curr_area / prev_area > 10 or prev_area / curr_area > 10:
                    to_delete.add(out_obj_id)
                    break
                
            if prev_detection is None and next_detection is None:
                frames[curr_frame_idx] = (None, 0, category,(0,0,0,0))
    for obj_id in to_delete:
        if obj_id in video_segments:
            del video_segments[obj_id]

    # merge segments
    video_segments = merge_centroids(video_segments)

    all_results = []
    for object_id, segment in video_segments.items():
        segments = extract_motion_segments(segment)
        # all movements
        results = determine_directions(segments, width, height, threshold, object_id)
        for _, result in results.items():
            if result['motion'] != 'no movement' or result['distance'] != 'no':
                all_results.append(results)
                break
    if len(all_results) != 0:
        return all_results
    
    return 'no'


if __name__ == '__main__':
    video_file_path = ''
    result = sam_generate_motion(video_file_path)
    print(result)
