# -- coding: utf-8 --

import numpy as np
import os
import six.moves.urllib as urllib
import urllib.request as allib
import sys
import tarfile
import tensorflow as tf
import time
import pytesseract
import pyttsx3
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append('/Users/gautamadhikari/Desktop/odfvip/models/research')

# Initialize pyttsx3 for voice feedback
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    """ Converts the provided text to speech immediately. """
    engine.say(text)
    engine.runAndWait()  # Blocks while processing

# Model settings
arch = 'resnet18'
model_file = f'whole_{arch}_places365_python36.pth.tar'
MODEL_NAME = 'ssd_inception_v2_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = '/opt/anaconda3/envs/objectdetection/bin/pytesseract'

# Import utils after setting paths
from object_detection.utils import label_map_util
from utils import visualization_utils as vis_util

# Load category index for label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Download model if not already present
if not os.path.exists(PATH_TO_CKPT):
    print('Downloading the model')
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())
    print('Download complete')
else:
    print('Model already exists')

# Load TensorFlow model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Set video capture (from mobile device)
#'http://192.168.71.99:8080/video' 
mobile_camera_url = 0 
cap = cv2.VideoCapture(mobile_camera_url)

width = 320  # Recommended width
height = 240  # Recommended height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Initialize last feedback time to prevent constant announcements
last_feedback_time = 0
feedback_interval = 5  # seconds between voice feedback

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        ret = True
        while ret:
            ret, image_np = cap.read()
            if not ret:
                print(f"Failed to capture image from mobile camera. Check your URL: {mobile_camera_url}")
                break

            image_np_expanded = np.expand_dims(image_np, axis=0)

            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded}
            )

            current_time = time.time()

            # Enhanced text rendering around detection boxes
            for i, b in enumerate(boxes[0]):
                if scores[0][i] >= 0.5:  # Only consider objects with a score greater than 0.5
                    mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
                    mid_y = (boxes[0][i][0] + boxes[0][i][2]) / 2
                    apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1])) ** 4), 1)

                    # Get object name from category index
                    object_name = category_index.get(classes[0][i], {}).get('name', 'Unknown object')

                    # Construct the feedback text
                    feedback_text = f"{object_name} detected at {apx_distance:.2f} meters."

                    # Set text position below the bounding box
                    org = (int(boxes[0][i][1] * image_np.shape[1]), int(boxes[0][i][2] * image_np.shape[0]))

                    # Enhanced text properties including background rectangle
                    text = f'{object_name} at {apx_distance:.2f} m'
                    text_org = (org[0], org[1] - 10)  # Position above the bounding box

                    # Font properties configuration
                    font_scale = 1.2  # Increase font size for better visibility
                    font_color = (255, 255, 255)
                    font_thickness = 3

                    # Compute sizes for text box
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

                    # Draw a filled rectangle as a background for the text
                    cv2.rectangle(image_np, (text_org[0], text_org[1] - text_height - baseline),
                                  (text_org[0] + text_width, text_org[1]), (0, 0, 0), thickness=cv2.FILLED)

                    # Draw the object name text on the image
                    cv2.putText(image_np, text, text_org, cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale, font_color, font_thickness, cv2.LINE_AA)

                    # Voice feedback with interval to avoid overlap
                    if (current_time - last_feedback_time) >= feedback_interval:
                        print(feedback_text)
                        speak(feedback_text)  # Call the speak function to convert text to speech
                        last_feedback_time = current_time  # Update feedback time

            # Visualization of detection boxes
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8
            )

            # Display frame with detections
            cv2.imshow('object detection', image_np)

            if cv2.waitKey(2) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()