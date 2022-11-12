import tensorflow as tf
import cv2
import numpy as np
import os
import shutil
import math
from PIL import Image

class InferFacePose:
    def __init__(self, model, im_size=192):
        self.im_size = im_size
        self.img_size = (im_size, im_size)
        self.model = model
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Mouth left corner
            (150.0, -150.0, -125.0)      # Mouth right corner
        ])
        self.focal_length = self.img_size[1]
        self.camera_center = (self.img_size[1] / 2, self.img_size[0] / 2)
        self.camera_matrix = np.array(
                    [[self.focal_length, 0, self.camera_center[0]],
                    [0, self.focal_length, self.camera_center[1]],
                    [0, 0, 1]], dtype="float")
        self.dist_coeffs = np.zeros((4, 1))

    def preprocess_img(self, img: Image.Image):
        img = cv2.resize(img, (self.im_size, self.im_size))
        img = img[...,::-1] # BGR2RGB
        img = img / 255.0
        return img

    def predict_is_mask_marks(self, img):
        img_batch = tf.expand_dims(img, 0)
        mask_prob, pred = self.model.predict(img_batch, verbose=0)
        mask_prob = np.array(mask_prob)[0][0]
        marks = np.array(pred[0]) * self.im_size
        return mask_prob, marks

    def get_3d_box(self, img, rotation_vector, translation_vector, camera_matrix, color=(255, 255, 0), line_width=2):
        point_3d = []
        dist_coeffs = np.zeros((4,1))
        rear_size = 1
        rear_depth = 0
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = img.shape[1]
        front_depth = front_size*2
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

        # Map to 2d img points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                        rotation_vector,
                                        translation_vector,
                                        camera_matrix,
                                        dist_coeffs)
        point_2d = np.int32(point_2d.reshape(-1, 2))
        
        k = (point_2d[5] + point_2d[8])//2
        
        return(point_2d[2], k)

    def euclide_dis(self, a, b):
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5 

    def get_dis_verti(self, ver_marks):
        res = []
        for i in range(len(ver_marks) - 1):
            res.append(self.euclide_dis(ver_marks[i], ver_marks[i + 1]))

        w = np.array([-6.16267425, 3.80783156, 0.93080629, -1.20422249, 0.24297246, 0.22946752, 0.60605209])
        res = np.array(res)
        res = np.sum(np.multiply(res, w))
        res += 1.5676940394777588

        return res

    def predict_is_mask_angle(self, img):
        """
        # mask_prob: [0,1] wearing mask prob
        # ang_vertical: x
        # ang_horizon: y
        # ang_rot: z
        # marks: (68, 2) marks of face pose
        """
        mask_prob, marks = self.predict_is_mask_marks(img)
        
        ver_marks = marks[[28,29,30,31,34,52,58,9]]
        key_marks = marks[[30,      # Nose tip
                            8,      # Chin
                            36,     # Left eye left corner
                            45,     # Right eye right corner
                            48,     # Left Mouth corner
                            54]     # Right mouth corner
                        ]

        dx = key_marks[2][1] - key_marks[3][1]
        dy = key_marks[2][0] - key_marks[3][0]

        ang_rot = np.arctan2(dy, dx)
        ang_rot = ang_rot * 180 / math.pi  

        (_, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points, 
                                                                key_marks, 
                                                                self.camera_matrix, 
                                                                self.dist_coeffs, 
                                                                flags=cv2.SOLVEPNP_UPNP)
        
        x1, x2 = self.get_3d_box(img, rotation_vector, translation_vector, self.camera_matrix)
            
        try:
            m = (x2[1] - x1[1]) / (x2[0] - x1[0])
            ang_horizon = math.degrees(math.atan(-1/m))
        except:
            ang_horizon = 90

        ang_vertical = self.get_dis_verti(ver_marks)

        ang_horizon = ang_horizon * -0.33112312 + -3.0374577755945733
        ang_rot = ang_rot * 0.90484725 + 81.45513723443987

        return mask_prob, ang_vertical, ang_horizon, ang_rot, marks

def set_memory_growth():
    physical_devices = tf.config.list_physical_devices('GPU') 
    for gpu_instance in physical_devices: 
        tf.config.experimental.set_memory_growth(gpu_instance, True)

def dummy_loss(y_true, y_pred):
    return 0.0

def draw_marks(image, marks, text=None, mark_size=3, color=(0, 255, 0), line_width=-1):
    """Draw the marks in image.
    Args:
        image: the image on which to be drawn.
        marks: points coordinates in a numpy array.
        mark_size: the size of the marks.
        color: the color of the marks, in BGR format, ranges 0~255.
        line_width: the width of the mark's outline. Set to -1 to fill it.
    """
    # We are drawing in an image, this is a 2D situation.
    image_copy = image.copy()
    for point in marks:
        cv2.circle(image_copy, (int(point[0]), int(point[1])),
                mark_size, color, line_width, cv2.LINE_AA)
    
    if text is not None:
        cv2.putText(image_copy,
                    text, 
                    (0,20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5,
                    (255,0,0),
                    1,
                    2)

    return image_copy




