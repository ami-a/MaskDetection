"""An example for using the TackEverything package
This example uses an Head Detection model from AVAuco/ssd_head_keras github repo
for detecting heads. It also uses a Face Mask classification model from
chandrikadeb7/Face-Mask-Detection github repo for the classification.
It can now easly detect, classify and track heads with/without masks in a video
using a few lines of code.
The use of the TrackEverything package make the models much more accurate
and robust, using tracking features and statistics.
"""
import os
import numpy as np
import cv2
#hide some tf loading data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# pylint: disable=wrong-import-position

from TrackEverything.detector import Detector
from TrackEverything.tool_box import DetectionVars,ClassificationVars,InspectorVars
from TrackEverything.statistical_methods import StatisticalCalculator, StatMethods
from TrackEverything.visualization_utils import VisualizationVars

import load_head_model

print("loading head detection model...")
det_model=load_head_model.get_head_model()
print("detection model loaded!")

#custome detection model interpolation
DETECTION_THRESHOLD=0.4
def custome_get_detection_array(
        image,
        detection_model=det_model,
        detection_threshold=DETECTION_THRESHOLD,
    ):
    """A method that utilize the detection model and return an np array of detections.
    Each detection in the format [confidence,(xmin,ymin,width,height)]
    Args:
        image (np.ndarray): current image
        detection_threshold (float): detection threshold
        model (tensorflow model obj): classification model
    """
    detections=detection_model.predict(np.asarray([image]))[0]
    num_detections = len(detections)
    #build the detection_array
    output= [
        [
            float(detections[i][1]),#score
            (
                max(0,int(detections[i][2])),#xmin
                max(0,int(detections[i][3])),#ymin
                min(int(detections[i][4])-int(detections[i][2]),image.shape[1]),#width
                min(int(detections[i][5])-int(detections[i][3]),image.shape[0]),#height
            )
        ]
            for i in range(num_detections) if
            detections[i][1]>detection_threshold #filter low detectin score
            and int(detections[i][4])-int(detections[i][2])>1
            and int(detections[i][5])-int(detections[i][3])>1
    ]
    #print(output)
    return output

#providing only the classification model path for ClassificationVars since the default loding method
#tf.keras.models.load_model(path) will work
CLASS_MODEL_PATH="classification_models/" \
"mask_class.model"
#custome classification model interpolation
def custome_classify_detection(model,det_images,size=(224,224)):
    """Classify a batch of images

    Args:
        model (tensorflow model): classification model
        det_images (np.array): batch of images in numpy array to classify
        size (tuple, optional): size to resize to, 1-D int32 Tensor of 2 elements:
            new_height, new_width (if None then no resizing).
            (In custome function you can use model.inputs[0].shape.as_list()
            and set size to default)
    Returns:
        Numpy NxM vector where N num of images, M num of classes and filled with scores.

        For example two images (car,plan) with three possible classes (car,plan,lion)
        that are identify currectly with 90% in the currect category and the rest is devided equally
        will return [[0.9,0.05,0.05],[0.05,0.9,0.05]].
    """
    #resize bounding box capture to fit classification model
    if size is not None:
        det_images=np.asarray(
            [
                cv2.resize(img, size, interpolation = cv2.INTER_LINEAR) for img in det_images
            ]
        )

    predictions=model.predict(det_images/255.)

    #if class is binary make sure size is 2
    if len(predictions)>0 and len(predictions[0])<2:
        reshaped_pred=np.ones((len(predictions),2))
        #size of classification list is 1 so turn it to 2
        for ind,pred in enumerate(predictions):
            reshaped_pred[ind,:]=pred,1-pred
        #print(reshaped_pred)
        predictions=reshaped_pred
    return predictions


#set the detector
detector_1=Detector(
    det_vars=DetectionVars(
        detection_model=det_model,
        detection_proccessing=custome_get_detection_array,
        detection_threshold=DETECTION_THRESHOLD
    ),
    class_vars=ClassificationVars(
        class_model_path=CLASS_MODEL_PATH,
        class_proccessing=custome_classify_detection
    ),
    inspector_vars=InspectorVars(
        stat_calc=StatisticalCalculator(method=StatMethods.FMA)
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
    dst_size=(512,512)

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
