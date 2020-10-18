"""Simple modul for performing video streaming/anlyzing
using the TrackEverything package
"""
import cv2

#play video with detector
def run_video(video_path,dst_size,detector):
    """play video with detector

    Args:
        video_path ([type]): [description]
        dst_size ([type]): [description]
    """
    cap = cv2.VideoCapture(video_path)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps=(cap.get(cv2.CAP_PROP_FPS))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"h:{height} w:{width} fps:{fps}")

    frame_number = -1
    while cap.isOpened():
        frame_number += 1
        ret, frame = cap.read()
        if not ret:
            break
        if frame_number%2==0:
            continue
        new_frm=frame
        #resize frame
        new_frm=cv2.resize(new_frm,dst_size,fx=0,fy=0, interpolation = cv2.INTER_LINEAR)
        #fix channel order since openCV flips them
        new_frm=cv2.cvtColor(new_frm, cv2.COLOR_BGR2RGB)

        #update the detector using the current frame
        detector.update(new_frm)
        #add the bounding boxes to the frame
        detector.draw_visualization(new_frm)

        #flip the channel order back
        new_frm=cv2.cvtColor(new_frm, cv2.COLOR_RGB2BGR)
        #resize frame
        new_frm=cv2.resize(new_frm,(width,height),fx=0,fy=0, interpolation = cv2.INTER_LINEAR)
        #show frame
        cv2.imshow('frame',new_frm)
        #get a small summary of the number of object of each class
        summ=detector.get_current_class_summary()
        print(f"{100*frame_number/length:.2f}%, frame:{frame_number}, summary:{summ}")
        #quite using the q key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
