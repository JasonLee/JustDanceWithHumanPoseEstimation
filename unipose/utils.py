import scipy.io as spio
import numpy as np


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def gaussian_heatmap(center=(2, 2), image_size=256, sig=1):
    """
    It produces single gaussian at expected center
    :param center:  the mean position (X, Y) - where high value expected
    :param image_size: The total image size (width, height)
    :param sig: The sigma value
    :return:
    """
    x_axis = np.linspace(0, image_size-1, image_size) - center[0]
    y_axis = np.linspace(0, image_size-1, image_size) - center[1]
    xx, yy = np.meshgrid(x_axis, y_axis)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return np.asfarray(kernel, float)

def evaluation_pckh(predicted_heatmaps, coords):
    PCKh = np.zeros((coords.shape[0] * coords.shape[1]))
    head_len = np.zeros(coords.shape[0])
    predicted_joints = local_maxima(predicted_heatmaps)

    # print("predicted_joints", predicted_joints)
    # print("coords", coords)
    # Batch Size
    for i in range(coords.shape[0]):
        if coords[i, 7, 0] == -1 or coords[i, 7, 1] == -1 or coords[i, 8, 0] == -1 or coords[i, 8, 1] == -1:
            print("There is an issue with head not recognised in dataset")
            continue

        head_len[i] = np.linalg.norm(coords[i, 7] - coords[i, 8])

        for j in range(coords.shape[1]):
            for k in range(1):
                if all(coords[i, j, k] == [-1, -1]):
                    continue

                dist = np.linalg.norm(predicted_joints[i, j, k] - coords[i, j, k])
                if dist <= head_len[i] * 0.5:
                    PCKh[i] = 1

    PCKh = np.mean(PCKh)

    return PCKh * 100

def local_maxima(heatmap):
    joint_coord = []

    for i in range(heatmap.shape[0]):
        arr = []
        for j in range(heatmap.shape[1]):
            arr.append(np.where(heatmap[i][j] == np.amax(heatmap[i][j])))

        joint_coord.append(arr)
        
    return np.array(joint_coord)
        

    



