import queue
import OpenPose.OpenPose

image_queue = queue_Queue()
    op = OpenPose()

    imageToProcess = cv2.imread("test.png")
    datum = op.process_image(imageToProcess)
    print("Body keypoints: \n" + str(datum.poseKeypoints))
    cv2.imshow("OpenPose 1.6.0 - Python API", datum.cvOutputData)
    cv2.waitKey(0)
