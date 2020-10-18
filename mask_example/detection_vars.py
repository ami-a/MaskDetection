"""loading the detection model variables for the detector object"""
import numpy as np
import cv2
from TrackEverything.tool_box import DetectionVars

import load_head_model
#detection variables
def get_det_vars_1():
    """loading the detection model variables for the detector object
    We define here the model interpolation function so the detector
    can use the model

    Returns:
        DetectionVars: the detection model variables
    """
    #custome loading the detection model and only providing the model to the DetectionVars
    print("loading head detection model...")
    det_model=load_head_model.get_head_model()
    print("detection model loaded!")

    #custome detection model interpolation
    detection_threshold=0.4
    def custome_get_detection_array(
            image,
            detection_model=det_model,
            detection_threshold=detection_threshold,
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

    return DetectionVars(
        detection_model=det_model,
        detection_proccessing=custome_get_detection_array,
        detection_threshold=detection_threshold
    )

def get_det_vars_2(prototxt_path, weights_path):
    """loading the detection model variables for the detector object
    We define here the model interpolation function so the detector
    can use the model

    Args:
        prototxt_path (str): The prototxt file path
        weights_path (str): The weights file path

    Returns:
        DetectionVars: the detection model variables
    """
    #custome loading the detection model and only providing the model to the DetectionVars
    #loading the detection model
    print("loading face detector model...")
    det_model = cv2.dnn.readNet(prototxt_path, weights_path)
    print("detection model loaded!")

    #custome detection model interpolation
    detection_threshold=0.12
    def custome_get_detection_array(
            image,
            detection_model=det_model,
            detection_threshold=detection_threshold,
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

    return DetectionVars(
        detection_model=det_model,
        detection_proccessing=custome_get_detection_array,
        detection_threshold=detection_threshold
    )
