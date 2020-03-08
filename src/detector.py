#!/usr/bin/env python

from __future__ import division

# Python imports
from utils.utils import *
from models import *
from torch.autograd import Variable
from torchvision import datasets
from torch.utils.data import DataLoader
import torch
import numpy as np
import scipy.io as sio
import os
import sys
import cv2
import time
from skimage.transform import resize

# ROS imports
import rospy
import std_msgs.msg
from rospkg import RosPack
from std_msgs.msg import UInt8, Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Polygon, Point32, Vector3
from yolov3_pytorch_ros.msg import BoundingBox, BoundingBoxes
from cv_bridge import CvBridge, CvBridgeError

package = RosPack()
package_path = package.get_path('yolov3_pytorch_ros')

# Deep learning imports


# Detector manager class for YOLO

class DetectorManager():
    def __init__(self):
        # Load weights parameter
        weights_name = rospy.get_param('~weights_name', 'yolov3.weights')
        self.weights_path = os.path.join(package_path, 'models', weights_name)
        rospy.loginfo("Found weights, loading %s", self.weights_path)

        # Raise error if it cannot find the model
        if not os.path.isfile(self.weights_path):
            raise IOError(('{:s} not found.').format(self.weights_path))

        # Load image parameter and confidence threshold
        self.image_topic = rospy.get_param(
            '~image_topic', '/camera/rgb/image_raw')
        self.confidence_th = rospy.get_param('~confidence', 0.7)
        self.nms_th = rospy.get_param('~nms_th', 0.3)

        # play video
        self.play_video = rospy.get_param('~play_video', False)

        # Load publisher topics
        self.published_image_topic = rospy.get_param('~detections_image_topic')
        self.light_size_topic = rospy.get_param(
            '~light_size_topic', "traffic_light_size")
        self.light_detection_topic = rospy.get_param(
            '~light_detection_topic', 'traffic_light_detected')

        # Load other parameters
        config_name = rospy.get_param('~config_name', 'yolov3.cfg')
        self.config_path = os.path.join(package_path, 'config', config_name)
        classes_name = rospy.get_param('~classes_name', 'coco.names')
        self.classes_path = os.path.join(package_path, 'classes', classes_name)
        self.gpu_id = rospy.get_param('~gpu_id', 0)
        self.network_img_size = rospy.get_param('~img_size', 608)
        self.publish_image = rospy.get_param('~publish_image')

        # Initialize width and height
        self.h = 0
        self.w = 0

        # Load net
        self.model = Darknet(self.config_path, img_size=self.network_img_size)
        self.model.load_weights(self.weights_path)
        if torch.cuda.is_available():
            self.model.cuda()
        else:
            raise IOError('CUDA not found.')
        self.model.eval()  # Set in evaluation mode
        rospy.loginfo("Deep neural network loaded")

        # Load CvBridge
        self.bridge = CvBridge()

        # Load classes
        # Extracts class labels from file
        self.classes = load_classes(self.classes_path)
        self.classes_colors = {}

        if(not self.play_video):
            self.image_sub = rospy.Subscriber(
                self.image_topic, Image, self.imageCb, queue_size=1, buff_size=2**24)

        # Define publishers
        self.light_size_pub = rospy.Publisher(
            self.light_size_topic, Vector3, queue_size=10)
        self.light_detection_pub = rospy.Publisher(
            self.light_detection_topic, Bool, queue_size=10)
        self.image_pub = rospy.Publisher(
            'test_image', Image, queue_size=10)

        rospy.loginfo("Launched node for object detection")

        if(self.play_video):
            self.video_location = rospy.get_param('~video_loc', '')
            self.rate_in_sec = rospy.get_param('~processing_rate', 1.0)
            rospy.timer(rospy.Duration(10), self.getVideo(
                self.video_location), True)

        # Spin
        rospy.spin()

    def imageCb(self, data):
        # Convert the image to OpenCV
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(e)

        self.detection(self.cv_image)

    def detection(self, image):
        # Configure input
        input_img = self.imagePreProcessing(image)
        input_img = Variable(input_img.type(torch.cuda.FloatTensor))

        # Get detections from network
        with torch.no_grad():
            detections = self.model(input_img)
            detections = non_max_suppression(
                detections, 80, self.confidence_th, self.nms_th)

        traff_light_size = Vector3()

        # Parse detections
        if detections[0] is not None:
            for detection in detections[0]:
                # Get xmin, ymin, xmax, ymax, confidence and class
                xmin, ymin, xmax, ymax, _, conf, det_class = detection
                pad_x = max(self.h - self.w, 0) * \
                    (self.network_img_size/max(self.h, self.w))
                pad_y = max(self.w - self.h, 0) * \
                    (self.network_img_size/max(self.h, self.w))
                unpad_h = self.network_img_size-pad_y
                unpad_w = self.network_img_size-pad_x
                xmin_unpad = ((xmin-pad_x//2)/unpad_w)*self.w
                xmax_unpad = ((xmax-xmin)/unpad_w)*self.w + xmin_unpad
                ymin_unpad = ((ymin-pad_y//2)/unpad_h)*self.h
                ymax_unpad = ((ymax-ymin)/unpad_h)*self.h + ymin_unpad

                # print(detection_msg.Class, "  ", int(det_class))
                if(det_class == 9):
                    traff_light_size.x = xmax_unpad - xmin_unpad
                    traff_light_size.y = ymax_unpad - ymin_unpad
                    traff_light_size.z = 0
                    self.light_detection_pub.publish(True)
                    self.light_size_pub.publish(traff_light_size)
                else:
                    self.light_detection_pub.publish(False)

        return True

    def getVideo(self, video_location):
        cap = cv2.VideoCapture(video_location)
        if not cap.isOpened():
            print "Error opening resource: " + str(video_location)
            exit(0)

        rval, frame = cap.read()
        # while rval:
        while rval:
            rval, frame = cap.read()
            image_message = self.bridge.cv2_to_imgmsg(
                frame, encoding="passthrough")
            self.image_pub.publish(image_message)
            self.detection(frame)
            rospy.sleep(self.rate_in_sec)

        cap.release()
        print("Processing video completed")

    def imagePreProcessing(self, img):
        # Extract image and shape
        img = np.copy(img)
        img = img.astype(float)
        height, width, channels = img.shape

        if (height != self.h) or (width != self.w):
            self.h = height
            self.w = width

            # Determine image to be used
            self.padded_image = np.zeros(
                (max(self.h, self.w), max(self.h, self.w), channels)).astype(float)

        # Add padding
        if (self.w > self.h):
            self.padded_image[(self.w-self.h)//2: self.h +
                              (self.w-self.h)//2, :, :] = img
        else:
            self.padded_image[:, (self.h-self.w) //
                              2: self.w + (self.h-self.w)//2, :] = img

        # Resize and normalize
        input_img = resize(
            self.padded_image, (self.network_img_size, self.network_img_size, 3))/255.

        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))

        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()
        input_img = input_img[None]

        return input_img

    def visualizeAndPublish(self, output, imgIn):
        # Copy image and visualize
        imgOut = imgIn.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.6
        thickness = 2
        for index in range(len(output.bounding_boxes)):
            label = output.bounding_boxes[index].Class
            x_p1 = output.bounding_boxes[index].xmin
            y_p1 = output.bounding_boxes[index].ymin
            x_p3 = output.bounding_boxes[index].xmax
            y_p3 = output.bounding_boxes[index].ymax
            confidence = output.bounding_boxes[index].probability
            w = int(x_p3 - x_p1)
            h = int(y_p3 - y_p1)
            # if w > h:
            #cv2.rectangle(imgOut, (int(x_p1)-20, int(y_p1)-20), (int(x_p3), int(y_p3)), (color[0],color[1],color[2]),2)

            print(w, h)

            # Find class color
            if label in self.classes_colors.keys():
                color = self.classes_colors[label]
            else:
                # Generate a new color if first time seen this label
                color = np.random.randint(0, 188, 3)
                #color[0] = (255,255,255)
                #color[1] = (198,198,198)
                #color[2] = (27,27,27)

                self.classes_colors[label] = color

            # Create rectangle
            cv2.rectangle(imgOut, (int(x_p1), int(y_p1)), (int(
                x_p3), int(y_p3)), (color[0], color[1], color[2]), 2)
            # print(color)
            text = ('{:s}: {:.3f}').format(label, confidence)
            cv2.putText(imgOut, text, (int(x_p1), int(y_p1-10)),
                        font, fontScale, (0, 0, 0), thickness, cv2.LINE_AA)

            # Create center
            center = (int(((x_p1)+(x_p3))/2), int(((y_p1)+(y_p3))/2))
            cv2.circle(imgOut, (center), 1, color, 3)
        # Publish visualization image
        image_msg = self.bridge.cv2_to_imgmsg(imgOut, "rgb8")
        self.pub_viz_.publish(image_msg)


if __name__ == "__main__":
    # Initialize node
    rospy.init_node("detector_manager_node")

    # Define detector object
    dm = DetectorManager()
