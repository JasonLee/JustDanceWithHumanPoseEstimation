import queue
from OpenPose import OpenPose
import cv2

PATH = "videos/dance_cover1.mp4"

def preprocess_video(path):
    op = OpenPose()
    cap = cv2.VideoCapture(path)
    while not cap.isOpened():
        cap = cv2.VideoCapture(path)
        cv2.waitKey(1000)
        print("Wait for the header")

    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    while True:
        flag, frame = cap.read()
        if flag:
            datum2 = op.process_image(frame)
            print(datum2.cvOutputData.shape)
            cv2.imshow("image", datum2.cvOutputData)
            # The frame is ready and already captured
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            print(str(pos_frame) + " frames")

        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
            print("frame is not ready")
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)

        if cv2.waitKey(10) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break

if __name__ == '__main__':
    preprocess_video(PATH)