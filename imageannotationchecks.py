
import requests
from requests.auth import HTTPBasicAuth
import json
import cv2
import math
import numpy as np
import csv

#LLMs have been used to help get this code running faster

def check_bounding_box_size(image, annotations_by_uuid):
    """
    Checks whether the bounding box for each annotation exceeds 20% of the total image area.

    Parameters:
        image
        annotations_by_uuid (dict): Dictionary mapping each uuid to its annotation details,
            where each detail includes 'width' and 'height' keys.
    
    Returns:
        probability of error value
        
    """

    image_height, image_width, _ = image.shape
    image_area = image_width * image_height
    psum = 0.0
    
    for uuid, annotation in annotations_by_uuid.items():
        bbox_width = annotation.get("width", 0)
        bbox_height = annotation.get("height", 0)
        bbox_area = bbox_width * bbox_height

        # Calculate the fraction of image area covered by the bounding box
        fraction = bbox_area / image_area
        if fraction > 0.20:
            psum+=0.2 #assigning 0.1 fraction per boundix box greater than 0.2

    return psum


def compute_iou(box1, box2):
    """
    Computes the Intersection over Union (IoU) between two bounding boxes.
    Each box is expected to be a dictionary with keys: 'left', 'top', 'width', 'height'.
    """
    # Convert each box to (x1, y1, x2, y2)
    x1_1 = box1["left"]
    y1_1 = box1["top"]
    x2_1 = x1_1 + box1["width"]
    y2_1 = y1_1 + box1["height"]

    x1_2 = box2["left"]
    y1_2 = box2["top"]
    x2_2 = x1_2 + box2["width"]
    y2_2 = y1_2 + box2["height"]

    # Compute the intersection coordinates
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    # Compute the area of intersection rectangle
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # Compute the area of both bounding boxes
    area1 = box1["width"] * box1["height"]
    area2 = box2["width"] * box2["height"]

    # Compute union area and then IoU
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    iou = inter_area / union_area
    return iou

def check_iou_annotations(img_cv, annotations_by_uuid, threshold=0.7):
    """
    Iterates through each pair of bounding boxes (annotations) and computes the IoU.
    
    Parameters:
      - img_cv: OpenCV image. 
      - annotations_by_uuid: Dictionary where keys are uuids and values are annotation dictionaries.
      - threshold: If specified, prints pairs with IoU above the threshold.
      
    Returns:
      A probability of error , PIOU value
    """
    # (Optional) Get image dimensions from img_cv.
    # For a color image, shape is (height, width, channels)
    height, width, _ = img_cv.shape
    print(f"Image dimensions: width={width}, height={height}")

    #iou_results = {}
    pIOU=0.0
    uuids = list(annotations_by_uuid.keys())
    for i in range(len(uuids)):
        for j in range(i + 1, len(uuids)):
            uuid1 = uuids[i]
            uuid2 = uuids[j]
            box1 = annotations_by_uuid[uuid1]
            box2 = annotations_by_uuid[uuid2]
            iou = compute_iou(box1, box2)
            #iou_results[(uuid1, uuid2)] = iou
            if iou > threshold:
                pIOU+=0.1
                #print(f"IOU more than threshold PIOU Value",pIOU)
    print("Bounding Box IOU Check complete")
    return pIOU


def check_color_consistency(img_cv, annotations_by_uuid):
    # Define background color options in BGR (OpenCV default)
    color_options = {
        "white":  (255, 255, 255),
        "yellow": (0, 255, 255),
        "red":    (0, 0, 255),
        "orange": (0, 165, 255),
        "green":  (0, 255, 0),
        "blue":   (255, 0, 0)
    }
    
    # Dictionary to group computed color classifications by annotation label
    classification_dict = {}
    
    # Process each annotation in the dictionary.
    for uuid, annotation in annotations_by_uuid.items():
        # Cast values to int
        left = int(annotation.get("left", 0))
        top = int(annotation.get("top", 0))
        width = int(annotation.get("width", 0))
        height = int(annotation.get("height", 0))
        
        # Crop the image region corresponding to the bounding box.
        bbox_region = img_cv[top:top+height, left:left+width]
        
        # If the bounding box is empty, skip this annotation.
        if bbox_region.size == 0:
            continue
        
        # Compute the average color in BGR.
        avg_color = np.mean(bbox_region, axis=(0, 1))
        avg_color = tuple(avg_color.astype(int))
        
        # Find the closest matching color option.
        best_match = None
        best_distance = float('inf')
        for color_name, color_value in color_options.items():
            distance = np.linalg.norm(np.array(avg_color) - np.array(color_value))
            if distance < best_distance:
                best_distance = distance
                best_match = color_name
        
        # Group the computed color by the annotation's "label".
        ann_label = annotation.get("label", "unknown")
        if ann_label not in classification_dict:
            classification_dict[ann_label] = []
        classification_dict[ann_label].append(best_match)
    
    # Count groups with inconsistent color classifications.
    count = 0.0
    for label, color_list in classification_dict.items():
        if len(set(color_list)) > 1:
            count += 1.0
    count = count / 20
    return count

def check_line_alignment(img_cv, annotations_by_uuid, threshold=15):
    """
    For each bounding box associated with each uuid in annotations_by_uuid, this function:
      - Extracts the bounding box region from img_cv.
      - Applies edge detection (Canny) and uses HoughLinesP to extract line segments.
      - For each detected line that is nearly horizontal or vertical (within angle tolerance),
        it computes the distance from the line's average coordinate to the corresponding bounding box boundary.
      - Chooses the candidate (the line closest to its parallel bounding box boundary) and if its distance exceeds
        the threshold (15 pixels by default), adds 0.1 to a counter.
      
    Parameters:
      img_cv: OpenCV image.
      annotations_by_uuid: Dictionary mapping each uuid to its annotation details.
                           Each annotation should include keys "left", "top", "width", "height".
      threshold: Distance threshold in pixels (default is 15).
      
    Returns:
      total_count: The final count (float) obtained by summing 0.1 for each bounding box whose best-aligned line
                   is farther than the threshold from its expected boundary.
    """
    total_count = 0.0
    angle_tolerance = 10  # degrees

    for uuid, annotation in annotations_by_uuid.items():
        # Cast values to int
        left = int(annotation.get("left", 0))
        top = int(annotation.get("top", 0))
        width = int(annotation.get("width", 0))
        height = int(annotation.get("height", 0))
        
        # Crop the bounding box region from the image.
        bbox_img = img_cv[top:top+height, left:left+width]
        if bbox_img.size == 0:
            continue
        
        # Convert crop to grayscale and detect edges.
        gray = cv2.cvtColor(bbox_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using the probabilistic Hough transform.
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                                minLineLength=min(width, height)/2, maxLineGap=10)
        
        # Define bounding box boundaries in the cropped coordinate system.
        bbox_left = 0
        bbox_top = 0
        bbox_right = width - 1
        bbox_bottom = height - 1
        
        candidate_distances = []
        
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    angle = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
                    avg_x = (x1 + x2) / 2.0
                    avg_y = (y1 + y2) / 2.0
                    
                    if abs(angle) <= angle_tolerance:
                        dist_top = abs(avg_y - bbox_top)
                        dist_bottom = abs(avg_y - bbox_bottom)
                        distance = min(dist_top, dist_bottom)
                        candidate_distances.append(distance)
                    elif abs(abs(angle) - 90) <= angle_tolerance:
                        dist_left = abs(avg_x - bbox_left)
                        dist_right = abs(avg_x - bbox_right)
                        distance = min(dist_left, dist_right)
                        candidate_distances.append(distance)
        
        if candidate_distances:
            closest_distance = min(candidate_distances)
            if closest_distance > threshold:
                total_count += 0.1

    return total_count

    
if __name__ == "__main__":
    #dummy urls are used below
    task_urls = [
        "url1",
        "url2",
        "url3",
        "url4",
        "url5",
        "url6",
        "url7",
        "url8",
    ]

    headers = {"Accept": "application/json"}
        
    auth = HTTPBasicAuth('Key', '') # fill key and password as needed
    with open("output.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header row
        writer.writerow(["url task string", "Error String", "probability of error (psum value)"])
    
        for url in task_urls:

        
            #fethcing the task response
            response = requests.request("GET", url, headers=headers, auth=auth)

            json_data = response.json()

            # Extract the .jpg image URL from the response
            attachment_url = json_data["params"]["attachment"]

            img_response = requests.get(attachment_url)
            img_array = np.asarray(bytearray(img_response.content), dtype=np.uint8)
            img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Assuming json_data is your JSON object
            annotations = json_data["response"]["annotations"]

            # Create a dictionary with uuid as key and annotation details as value
            annotations_by_uuid = { annotation["uuid"]: annotation for annotation in annotations }

            p_size = check_bounding_box_size(img_cv, annotations_by_uuid)
            print("Bounding box error fraction",p_size)

            p_IOU = check_iou_annotations(img_cv, annotations_by_uuid, threshold=0.8)

            p_color = check_color_consistency(img_cv,annotations_by_uuid)

            print('Color consistency Count', p_color)

            p_line = check_line_alignment(img_cv,annotations_by_uuid)
            print('Line alignment Vlaue', p_line)

            p_sum = p_size+p_IOU+p_color+p_line

            if p_sum >=0.2:
              result_str = "Error"
            elif p_sum <0.2 and p_sum>=0.1:
                result_str = "Warning"
            else:
                result_str = "All good"
            
            print("1 more image result")
            print(result_str)
            # Write a row to the CSV file
            writer.writerow([url, result_str, p_sum])


