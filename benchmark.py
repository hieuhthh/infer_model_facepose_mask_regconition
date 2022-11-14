import os
import cv2
import numpy as np
import pandas as pd

from facepose import *

from crop_face import *

os.environ["CUDA_VISIBLE_DEVICES"]="3"

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

route = 'dataset'
out_route = 'benchmark'

try:
    os.mkdir(out_route)
except:
    pass

x_dis = []
y_dis = []
z_dis = []
dists = []

for file in os.listdir(route):
    file_path = os.path.join(route, file)
    img = cv2.imread(file_path) # BGR here

    bb_box = extract_face(img)
    img = crop_face(img, bb_box)

    img = infer_face_pose.preprocess_img(img) # BGR2RGB, Resize, /255

    mask_prob, ang_vertical, ang_horizon, ang_rot, marks = infer_face_pose.predict_is_mask_angle(img)

    # mask_prob: [0,1] wearing mask prob
    # ang_vertical: x
    # ang_horizon: y
    # ang_rot: z
    # marks: (68, 2) marks of face pose

    img_write = np.array(img)[...,::-1] * 255
    # class_text = f'mask prob: {round(mask_prob,2)}'
    class_text = f'{round(mask_prob,2)} {int(ang_vertical)} {int(ang_horizon)} {int(ang_rot)}'
    img_write = draw_marks(img_write, marks, class_text)
    cv2.imwrite(f"./{out_route}/{file}", img_write)

    # print("Name, x, y, z:", file, ang_vertical, ang_horizon, ang_rot)

    x_dis.append(abs(ang_vertical))
    y_dis.append(abs(ang_horizon))
    z_dis.append(abs(ang_rot))

    dists.append((abs(ang_vertical) + abs(ang_horizon) + abs(ang_rot)) / 3)

def describe(arrs, names):
    with open('benchmark.txt', 'w') as f:
        f.write('name,mean,median,min,max,range,variance,sd,per25,per50,per75\n')
        print(('name,mean,median,min,max,range,variance,sd,per25,per50,per75\n'))

        for i, arr in enumerate(arrs):
            _name = names[i]
            _mean = np.mean(arr)
            _median = np.median(arr)
            _min = np.amin(arr)
            _max = np.amax(arr)
            _range = np.ptp(arr)
            _variance = np.var(arr)
            _sd = np.std(arr)
            _per25 = np.percentile(arr, 25)
            _per50 = np.percentile(arr, 50)
            _per75 = np.percentile(arr, 75)

            f.write(f'{_name},{_mean},{_median},{_min},{_max},{_range},{_variance},{_sd},{_per25},{_per50},{_per75}\n')
            print(f'{_name},{_mean},{_median},{_min},{_max},{_range},{_variance},{_sd},{_per25},{_per50},{_per75}\n')
            
list_des = [x_dis, y_dis, z_dis, dists]
names = ['x', 'y', 'z', 'all']
describe(list_des, names)