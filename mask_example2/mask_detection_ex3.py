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
#hide some tf loading data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# pylint: disable=wrong-import-position
from TrackEverything.detector import Detector
from TrackEverything.tool_box import InspectorVars
from TrackEverything.statistical_methods import StatisticalCalculator, StatMethods
from TrackEverything.visualization_utils import VisualizationVars

from detection_vars import get_det_vars
from play_video import run_video
# pylint: enable=wrong-import-position

DET_MODEL_PATH="detection_models/mask_ckp/mask_weights_epoch_100.h5"

#not providing a classification vars since the detection model also classifies

#set the detector
detector_1=Detector(
    det_vars=get_det_vars(DET_MODEL_PATH),
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
#since the head detction model requires a (320,240) image input
run_video(VIDEO_PATH,(320,240),detector_1)
# from play_video import save_video
# save_video(VIDEO_PATH,(320,240),detector_1,"video/mask_03.avi")
