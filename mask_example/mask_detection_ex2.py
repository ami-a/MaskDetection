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
#hide some tf loading data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# pylint: disable=wrong-import-position
from TrackEverything.detector import Detector
from TrackEverything.tool_box import InspectorVars
from TrackEverything.statistical_methods import StatisticalCalculator, StatMethods
from TrackEverything.visualization_utils import VisualizationVars

from detection_vars import get_det_vars_2
from classification_vars import get_class_vars
from play_video import run_video
# pylint: enable=wrong-import-position

PROTOTXT_PATH = "detection_models/face_detector/deploy.prototxt"
WEIGHTS_PATH = "detection_models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"

CLASS_MODEL_PATH="classification_models/" \
"mask_class.model"

#set the detector
detector_1=Detector(
    det_vars=get_det_vars_2(PROTOTXT_PATH, WEIGHTS_PATH),
    class_vars=get_class_vars(CLASS_MODEL_PATH),
    inspector_vars=InspectorVars(
        stat_calc=StatisticalCalculator(method=StatMethods.Non)
    ),
    visualization_vars=VisualizationVars(
        labels=["No Mask","Mask"],
        colors=["Red","Green","Cyan"],#last color for trackers
        show_trackers=True,
        uncertainty_threshold=0.1,
        uncertainty_label="Collecting Info"
    )
)

#Test it on a video
VIDEO_PATH="video/OxfordStreet.mp4"
#since the head detction model requires a 512x512 image input
run_video(VIDEO_PATH,(512,512),detector_1)
# from play_video import save_video
# save_video(VIDEO_PATH,(512,512),detector_1,"video/mask_02.avi")
