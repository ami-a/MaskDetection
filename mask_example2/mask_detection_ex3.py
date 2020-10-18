"""An example for using the TackEverything package
This example uses a Mask Detection and Classification
model from PureHing/face-mask-detection-tf2 github repo
for detecting faces and classify them. Here there is no need
for a sprate classification model. It can now easly detect, classify
and track heads with/without masks in a video using a few lines of code.
The use of the TrackEverything package make the model much more accurate
and robust, using tracking features and statistics.
"""
import os
import numpy as np
import cv2
#hide some tf loading data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# pylint: disable=wrong-import-position
import tensorflow as tf
from TrackEverything.detector import Detector
from TrackEverything.tool_box import DetectionVars,InspectorVars
from TrackEverything.statistical_methods import StatisticalCalculator, StatMethods
from TrackEverything.visualization_utils import VisualizationVars

from components import config
from components.prior_box import priors_box
from components.utils import decode_bbox_tf, compute_nms
from network.network import SlimModel

print("loading head detection model...")
cfg = config.cfg
image_size=(240, 320)
priors, num_cell = priors_box(cfg, image_size)
priors = tf.cast(priors, tf.float32)
DET_MODEL_PATH="detection_models/mask_ckp/mask_weights_epoch_100.h5"
det_model = SlimModel(cfg=cfg, num_cell=num_cell, training=False)
det_model.load_weights(DET_MODEL_PATH)
print("detection model loaded!")

#custome detection model interpolation
DETECTION_THRESHOLD=0.4
def get_box_cordinates(box,img_shape):
    """#convert model cordinates format to (xmin,ymin,width,height)

    Args:
        box ((xmin,xmax,ymin,ymax)): the cordinates are relative [0,1]
        img_shape ((height,width,channels)): the frame size

    Returns:
        (xmin,ymin,width,height): (xmin,ymin,width,height): converted cordinates
    """
    height,width, = img_shape[:2]
    xmin=max(int(box[0]*width),0)
    ymin=max(0,int(box[1]*height))
    xmax=min(int(box[2]*width),width-1)
    ymax=min(int(box[3]*height),height-1)
    return (
        xmin,#xmin
        ymin,#ymin
        xmax-xmin,#box width
        ymax-ymin#box height
    )

def parse_predict(predictions):
    """Interpreting the predictions to score classes and boxes.

    Args:
        predictions (np.array): [description]contain all the detections info of a frame.

    Returns:
        [boxes, classes, scores]: the predictions boxes, classes & scores.
    """
    label_classes = cfg['labels_list']

    bbox_regressions, confs = tf.split(predictions[0], [4, -1],-1)
    boxes = decode_bbox_tf(bbox_regressions, priors, cfg['variances'])
    ##classifications shape :(num_priors,num_classes)

    confs = tf.math.softmax(confs, axis=-1)

    out_boxes = []
    out_labels = []
    out_scores = []

    for i in range(1, len(label_classes)):
        cls_scores = confs[:, i]

        score_idx = cls_scores > DETECTION_THRESHOLD

        cls_boxes = boxes[score_idx]
        cls_scores = cls_scores[score_idx]

        nms_idx = compute_nms(cls_boxes, cls_scores, cfg['nms_threshold'], cfg['max_number_keep'])

        cls_boxes = tf.gather(cls_boxes, nms_idx, axis=None)
        cls_scores = tf.gather(cls_scores, nms_idx, axis=None)

        cls_labels = [i] * cls_boxes.shape[0]

        out_boxes.append(cls_boxes)
        out_labels.extend(cls_labels)
        out_scores.append(cls_scores)

    out_boxes = tf.concat(out_boxes, 0)
    out_scores = tf.concat(out_scores, 0)

    boxes = tf.clip_by_value(out_boxes, 0.0, 1.0).numpy()
    classes = np.array(out_labels)

    return boxes, classes,  out_scores.numpy()

def custome_get_detection_array(
        image,
        detection_model=det_model,
    ):
    """A method that utilize the detection model and return an np array of detections.
    Each detection in the format [confidence,(xmin,ymin,width,height)]
    Args:
        image (np.ndarray): current image
        detection_threshold (float): detection threshold
        model (tensorflow model obj): classification model
    """
    image=(image / 255.0 - 0.5) / 1.0
    detections=detection_model.predict(image[np.newaxis, ...])
    boxes, classes, scores=parse_predict(detections)
    num_detections = len(boxes)

    #build the detection_array
    output= [
        [
            get_class(scores[i],classes[i]),#score
            get_box_cordinates(boxes[i],image.shape),
        ]
            for i in range(num_detections)
    ]
    #print(output)
    return output

def get_class(score,cls):
    """Convert a score a 2 classes index
    to a 2D score vector

    Args:
        score (float): The total score of cls+detection

    Returns:
        np.array: the score devided between two array position with the cls index
        having the score value and the other one 1-score
    """
    res=np.zeros((2))
    idx=cls % 2
    res[idx]=score
    res[(idx+1)%2]=1-score
    return res

#not providing a classification model since the detection model also classifies

#set the detector
detector_1=Detector(
    det_vars=DetectionVars(
        detection_model=det_model,
        detection_proccessing=custome_get_detection_array,
        detection_threshold=DETECTION_THRESHOLD
    ),
    inspector_vars=InspectorVars(
        stat_calc=StatisticalCalculator(method=StatMethods.EMA)
    ),
    visualization_vars=VisualizationVars(
        labels=["No Mask","Mask"],
        colors=["Red", "Green","Cyan"],#last color for trackers
        show_trackers=True,
        uncertainty_threshold=0.5,
        uncertainty_label="Getting Info"
    )
)

#Test it on a video
VIDEO_PATH="video/OxfordStreet.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps=(cap.get(cv2.CAP_PROP_FPS))
h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(f"h:{h} w:{w} fps:{fps}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

TRIM=True#whether or not to resize frame
#since the head detction model requires a 512x512 image input
if TRIM and w!=512 or h!=512:
    dst_size=(image_size[1],image_size[0])

FRAME_NUMBER = -1
while cap.isOpened():
    FRAME_NUMBER += 1
    ret, frame = cap.read()
    if not ret:
        break
    new_frm=frame
    if TRIM:
        #resize frame
        new_frm=cv2.resize(new_frm,dst_size,fx=0,fy=0, interpolation = cv2.INTER_LINEAR)
    #fix channel order since openCV flips them
    new_frm=cv2.cvtColor(new_frm, cv2.COLOR_BGR2RGB)

    #update the detector using the current frame
    detector_1.update(new_frm)
    #add the bounding boxes to the frame
    detector_1.draw_visualization(new_frm)

    #flip the channel order back
    new_frm=cv2.cvtColor(new_frm, cv2.COLOR_RGB2BGR)
    if TRIM:
        #resize frame
        new_frm=cv2.resize(new_frm,(w,h),fx=0,fy=0, interpolation = cv2.INTER_LINEAR)
    #show frame
    cv2.imshow('frame',new_frm)
    #get a small summary of the number of object of each class
    summ=detector_1.get_current_class_summary()
    print(f"frame:{FRAME_NUMBER}, summary:{summ}")
    #quite using the q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
