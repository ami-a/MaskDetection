"""A module for loading the  Head Detection model from AVAuco/ssd_head_keras github repo
"""
from keras import backend as K
from keras.optimizers import Adam
from models.keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss

def get_head_model():
    """A method for loading the Head Detection model from AVAuco/ssd_head_keras github repo

    Returns:
        tf modle: Head Detection model
    """
    # Set the image size.
    img_height = 512
    img_width = 512
    # Set the model's inference mode
    model_mode = 'inference'


    # Set the desired confidence threshold
    conf_thresh = 0.01

    # 1: Build the Keras model
    K.clear_session() # Clear previous models from memory.
    model = ssd_512(image_size=(img_height, img_width, 3),
                    n_classes=1,
                    mode=model_mode,
                    l2_regularization=0.0005,
                    scales=[0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05], # PASCAL VOC
                    aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                            [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                            [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                            [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                            [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                            [1.0, 2.0, 0.5],
                                            [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 128, 256, 512],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=conf_thresh,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)

    # 2: Load the trained weights into the model.
    weights_path = './detection_models/head_detection_ssd512-hollywood-trainval.h5'
    model.load_weights(weights_path, by_name=True)

    # 3: Compile the model so that Keras won't complain the next time you load it.
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    # model.save('./detection_models/head_detection')
    return model
