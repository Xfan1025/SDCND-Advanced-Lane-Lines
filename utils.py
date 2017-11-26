import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob



class Line(object):
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False

        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
    #         #polynomial coefficients for the most recent fit
    #         self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None


        # peak detected
        self.peak = None

        self.fitx = None

def get_camera_calib():
    '''camera calibration -> (mtx, dist)'''
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane. -- corner points

    # load calibration images
    camera_cal_dir = glob.glob('./camera_cal/*.jpg')

    for img_path in camera_cal_dir:
        img = mpimg.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        
        # append points found into the list
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            
            # draw and display the corners
            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            
    # Do camera calibration given object points and image points
    # shape can only be 2d: 
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist

def get_perspective():
    ''' Perspective transformation -> (M, Minv) '''
    # read in an image with stright road to calculate the perspective transform
    straight_road = mpimg.imread('./test_images/straight_lines1.jpg')

    # define 4 source points
    ### top left - bottom left - bottom right - top right
    src = np.float32([[592, 453],
                      [202, 720],
                      [1100, 720],
                      [685, 453]])
    # define 4 destination points
    dst = np.float32([[360, 0],
                      [360, 720],
                      [970, 720],
                      [970, 0]])




    # compute the perspective transform M and reversed transform Minv
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


# def get_offset():


