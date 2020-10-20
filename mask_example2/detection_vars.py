"""loading the detection model variables for the detector object"""
import tensorflow as tf
import numpy as np
from TrackEverything.tool_box import DetectionVars
from components import config
from components.prior_box import priors_box
from components.utils import decode_bbox_tf, compute_nms
from network.network import SlimModel
#initial variables
cfg = config.cfg
image_size=(240, 320)
priors, num_cell = priors_box(cfg, image_size)
priors = tf.cast(priors, tf.float32)
#detection variables
def get_model(det_model_path):
    """Get the model obj

    Args:
        det_model_path (tf.model): path to model

    Returns:
        [type]: [description]
    """
    #loading the detection model
    print("loading head detection model...")
    det_model = SlimModel(cfg=cfg, num_cell=num_cell, training=False)
    det_model.load_weights(det_model_path)
    print("detection model loaded!")
    return det_model
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

        nms_idx = compute_nms(
            cls_boxes,
            cls_scores,
            cfg['nms_threshold'],
            cfg['max_number_keep']
        )

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
        detection_var,
        image,
    ):
    """A method that utilize the detection model and return an np array of detections.
    Each detection in the format [confidence,(xmin,ymin,width,height)]
    Args:
        detection_var (DetectionVars): the DetectionVars obj
        image (np.ndarray): current image
        detection_threshold (float): detection threshold
    """
    detection_model=detection_var.detection_model
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

def get_det_vars(det_model_path):
    """loading the detection model variables for the detector object
    We define here the model interpolation function so the detector
    can use the model

    Args:
        det_model_path (str): The model path

    Returns:
        DetectionVars: the detection model variables
    """
    return DetectionVars(
        detection_model=get_model(det_model_path),
        detection_proccessing=custome_get_detection_array,
        detection_threshold=DETECTION_THRESHOLD
    )
