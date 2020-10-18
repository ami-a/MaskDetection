"""An example for using the TackEverything package
This example uses a Face Detection model from OpenCv
for detecting faces. It also uses a Face Mask classification model from
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


print("loading face detector model...")
PROTOTXT_PATH = "detection_models/face_detector/deploy.prototxt"
WEIGHTS_PATH = "detection_models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
det_model = cv2.dnn.readNet(PROTOTXT_PATH, WEIGHTS_PATH)
print("detection model loaded!")

#custome detection model interpolation
DETECTION_THRESHOLD=0.12
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
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0),swapRB=False)

    # pass the blob through the network and obtain the face detections
    detection_model.setInput(blob)
    detections = detection_model.forward()[0,0]
    #detections=detection_model.predict(np.asarray([image]))[0]
    num_detections = len(detections)
    #boxes
    boxes =[get_box_cordinates(detections[i][3:7],image.shape) for i in range(num_detections)]
    #build the detection_array
    output= [
        [
            float(detections[i][2]),#score
            boxes[i],
        ]
            for i in range(num_detections) if
            detections[i][2]>detection_threshold #filter low detectin score
            and min(boxes[i][2:])>0
    ]
    #print(output)
    return output

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
    # for img in det_images:
    #     print(img.shape)
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
        labels=["Mask","No Mask"],
        colors=[ "Green","Red","Cyan"],#last color for trackers
        show_trackers=True,
        uncertainty_threshold=0.2,
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
    if FRAME_NUMBER%2==0:
        continue
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
