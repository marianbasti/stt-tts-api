# This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# For more details, visit: https://creativecommons.org/licenses/by-nc/4.0/
# 
# # livelink_init.py

import socket
from utils.livelink.connect.pylivelinkface import PyLiveLinkFace, FaceBlendShape
import logging

logging.basicConfig(level=logging.INFO)

UDP_IP = "127.0.0.1"
UDP_PORT = 11111

def create_socket_connection():
    try:
        logging.info(f"Creating socket connection to {UDP_IP}:{UDP_PORT}")
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((UDP_IP, UDP_PORT))
        logging.info("Socket connection established")
        return s
    except socket.error as e:
        logging.error(f"Failed to create socket connection: {e}")
        return None

def initialize_py_face():
    try:
        logging.info("Initializing PyLiveLinkFace")
        py_face = PyLiveLinkFace()
        initial_blendshapes = [0.0] * 61
        for i, value in enumerate(initial_blendshapes):
            py_face.set_blendshape(FaceBlendShape(i), float(value))
        logging.info("PyLiveLinkFace initialized with default blendshapes")
        return py_face
    except Exception as e:
        logging.error(f"Failed to initialize PyLiveLinkFace: {e}")
        return None