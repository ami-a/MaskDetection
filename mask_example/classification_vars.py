"""loading the classification model variables for the detector object"""
import numpy as np
import cv2
from TrackEverything.tool_box import ClassificationVars
def get_class_vars(class_model_path):
    """loading the classification model variables for the detector object
    We define here the model interpolation function so the detector
    can use the classification model

    Args:
        class_model_path (str): classification model path

    Returns:
        ClassificationVars: classification variables for the detector
    """
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
            that are identify currectly with 90% in the currect category and the rest is
            devided equally will return [[0.9,0.05,0.05],[0.05,0.9,0.05]].
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

    #providing only the classification model path for ClassificationVars
    #since the default loding method
    #tf.keras.models.load_model(path) will work
    return ClassificationVars(
        class_model_path=class_model_path,
        class_proccessing=custome_classify_detection
    )
