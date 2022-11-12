import os
import cv2
import numpy as np
import pandas as pd

from facepose import *

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# optional
set_memory_growth() 

im_size = 192
weight = 'best_model_facepose_with_mask_EfficientNetV2S_192_512_2.h5'
# https://drive.google.com/file/d/1-aja0jCAdiFclVx_WDjJeUHdiHHBwM-J/view?usp=share_link

strategy = tf.distribute.get_strategy()
with strategy.scope():
    model = tf.keras.models.load_model(weight, custom_objects={'dummy_loss':dummy_loss})
    model.summary()

infer_face_pose = InferFacePose(model, im_size)

route = 'sample'
out_route = 'predict'

try:
    os.mkdir(out_route)
except:
    pass

for file in os.listdir(route):
    file_path = os.path.join(route, file)
    img = cv2.imread(file_path) # BGR here
    img = infer_face_pose.preprocess_img(img) # BGR2RGB, Resize, /255

    mask_prob, ang_vertical, ang_horizon, ang_rot, marks = infer_face_pose.predict_is_mask_angle(img)

    # mask_prob: [0,1] wearing mask prob
    # ang_vertical: x
    # ang_horizon: y
    # ang_rot: z
    # marks: (68, 2) marks of face pose

    img_write = np.array(img)[...,::-1] * 255
    class_text = f'mask prob: {round(mask_prob,2)}'
    img_write = draw_marks(img_write, marks, class_text)
    cv2.imwrite(f"./{out_route}/{file}", img_write)

    print("Name, x, y, z:", file, ang_vertical, ang_horizon, ang_rot)