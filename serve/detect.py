import argparse
import requests
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
import pprint


def make_request(url, image_path):
    headers = {
        'Content-Type': 'image/jpeg'
    }
    
    with open(image_path, "rb") as image:
        f = image.read()
        payload = bytearray(f)
        
    response = requests.request("POST", url, headers=headers, data=payload)
    return json.loads(response.text)


def visualize_detections(image_path, detections, result_image_path, figsize=(8, 8)):
    img = Image.open(image_path)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(img)

    img_width, img_height = img.size  # Get the original image dimensions

    scoreArr, nameArr, boxArr = [], [], []

    for detection in detections:
        score = detection['confidence']
        # score = detection['conf']
        name = detection['class']  # class_names
        # name = detection['label']  # class_names
        box = [detection['x1'], detection['y1'], detection['x2'], detection['y2']]  # boxes
        scoreArr.append(score)
        nameArr.append(name)
        boxArr.append(box)

    scoreArr, nameArr, boxArr = np.array(scoreArr), np.array(nameArr), np.array(boxArr)

    boxes, class_names, scores = boxArr, nameArr, scoreArr
    max_boxes, min_score = 18, 0.1

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            xmin, ymin, xmax, ymax = tuple(boxes[i])

            ax = plt.gca()

            # Scale the bounding box coordinates based on the original image dimensions
            xmin *= img_width
            xmax *= img_width
            ymin *= img_height
            ymax *= img_height

            w, h = xmax - xmin, ymax - ymin

            if class_names[i] == 'hat':
                patch = plt.Rectangle(
                    [xmin, ymin], w, h, fill=False, edgecolor='c', linewidth=3
                )
            else:
                patch = plt.Rectangle(
                    [xmin, ymin], w, h, fill=False, edgecolor='r', linewidth=3
                )

            ax.add_patch(patch)

    ax.text(
        img_width * 0.5,  # Adjust the horizontal position
        img_height * 0.95,  # Adjust the vertical position
        # "蓝框: 戴安全帽; 红框: 不戴安全帽",
        "Blue: Wear Helmet; Red: No Helmet",
        color='white',
        backgroundcolor='black',
        fontsize=14,
    )

    plt.savefig(result_image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='test_1.jpg', help='input image') 
    parser.add_argument('--output', type=str, default='result_test_1.jpg', help='output image') 
    opt = parser.parse_args()
    print(opt)

    url="http://localhost:8080/predictions/helmet_detection"
    
    # image_path = opt.input
    # result_image_path = opt.output
    
    detections = make_request(url, opt.input)
    
    print('\nDetecting ...\n')
    pprint.pprint(detections)
    
    print('\nLabelling ...\n')
    visualize_detections(opt.input, detections, opt.output)
    
    print('Result saved into', opt.output)
