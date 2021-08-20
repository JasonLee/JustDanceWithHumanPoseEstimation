# PoseEstimation Interface
class PoseEstimation(object):
    def __init__(self):
        pass

    def show(self):
        raise Exception("NotImplementedException")

    # returns joint coords
    def process_image_keypoints(self, image):
        raise Exception("NotImplementedException")

    # returns how joints are connected
    def get_joint_mapping(self):
        raise Exception("NotImplementedException")

    # return number of joints
    def get_length(self):
        raise Exception("NotImplementedException")

    def get_cv_image(self, image):
        raise Exception("NotImplementedException")