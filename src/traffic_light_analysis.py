#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Vector3


class Analysis:
    def __init__(self):
        self.analysis_topic_pub = rospy.Publisher(
            'analysis_topic', Float32, queue_size=10)
        self.light_detection_sub = rospy.Subscriber(
            'detection_topic', Bool, self.detection_cb, queue_size=1)
        self.light_size_sub = rospy.Subscriber(
            'size_topic', Vector3, self.size_cb, queue_size=1)

        self.light_detected = False

    def detection_cb(self, msg):
        self.light_detected = msg.data

    def size_cb(self, msg):
        if self.light_detected:
            zone_height = Float32()
            zone_height = msg.y / 3.0
            self.analysis_topic_pub.publish(zone_height)
        self.light_detected = False


if __name__ == "__main__":
    rospy.init_node("analysis_node")

    Analysis()
    rospy.spin()
