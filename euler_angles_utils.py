import cv2
import numpy as np
import math

def calculate_pitch_yaw_roll(landmarks_2D ,cam_w=256, cam_h=256,radians=False):
    """ Return the the pitch  yaw and roll angles associated with the input image.
    @param radians When True it returns the angle in radians, otherwise in degrees.
    """
    c_x = cam_w/2
    c_y = cam_h/2
    f_x = c_x / np.tan(60/2 * np.pi / 180)
    f_y = f_x

    #Estimated camera matrix values.
    camera_matrix = np.float32([[f_x, 0.0, c_x],
                                [0.0, f_y, c_y],
                                [0.0, 0.0, 1.0]])

    camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

    #The dlib shape predictor returns 68 points, we are interested only in a few of those
    # TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
    #wflw(98 landmark) trached points
    # TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
    #X-Y-Z with X pointing forward and Y on the left and Z up.
    #The X-Y-Z coordinates used are like the standard
    # coordinates of ROS (robotic operative system)
    #OpenCV uses the reference usually used in computer vision:
    #X points to the right, Y down, Z to the front
    LEFT_EYEBROW_LEFT  = [6.825897, 6.760612, 4.402142]
    LEFT_EYEBROW_RIGHT = [1.330353, 7.122144, 6.903745]
    RIGHT_EYEBROW_LEFT = [-1.330353, 7.122144, 6.903745]
    RIGHT_EYEBROW_RIGHT= [-6.825897, 6.760612, 4.402142]
    LEFT_EYE_LEFT  = [5.311432, 5.485328, 3.987654]
    LEFT_EYE_RIGHT = [1.789930, 5.393625, 4.413414]
    RIGHT_EYE_LEFT = [-1.789930, 5.393625, 4.413414]
    RIGHT_EYE_RIGHT= [-5.311432, 5.485328, 3.987654]
    NOSE_LEFT  = [2.005628, 1.409845, 6.165652]
    NOSE_RIGHT = [-2.005628, 1.409845, 6.165652]
    MOUTH_LEFT = [2.774015, -2.080775, 5.048531]
    MOUTH_RIGHT=[-2.774015, -2.080775, 5.048531]
    LOWER_LIP= [0.000000, -3.116408, 6.097667]
    CHIN     = [0.000000, -7.415691, 4.070434]

    landmarks_3D = np.float32( [LEFT_EYEBROW_LEFT,
                                LEFT_EYEBROW_RIGHT,
                                RIGHT_EYEBROW_LEFT,
                                RIGHT_EYEBROW_RIGHT,
                                LEFT_EYE_LEFT,
                                LEFT_EYE_RIGHT,
                                RIGHT_EYE_LEFT,
                                RIGHT_EYE_RIGHT,
                                NOSE_LEFT,
                                NOSE_RIGHT,
                                MOUTH_LEFT,
                                MOUTH_RIGHT,
                                LOWER_LIP,
                                CHIN])

    #Return the 2D position of our landmarks
    assert landmarks_2D is not None ,'landmarks_2D is None'
    landmarks_2D = np.asarray(landmarks_2D,dtype=np.float32).reshape(-1,2)
    #Applying the PnP solver to find the 3D pose
    #of the head from the 2D position of the
    #landmarks.
    #retval - bool
    #rvec - Output rotation vector that, together with tvec, brings
    #points from the world coordinate system to the camera coordinate system.
    #tvec - Output translation vector. It is the position of the world origin (SELLION) in camera co-ords
    retval, rvec, tvec = cv2.solvePnP(landmarks_3D,
                                      landmarks_2D,
                                      camera_matrix,
                                      camera_distortion)

    #Get as input the rotational vector
    #Return a rotational matrix
    rmat, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat((rmat,tvec))

    #euler_angles contain (pitch, yaw, roll)
    # euler_angles = cv2.DecomposeProjectionMatrix(projMatrix=rmat, cameraMatrix=self.camera_matrix, rotMatrix, transVect, rotMatrX=None, rotMatrY=None, rotMatrZ=None)
    _, _, _, _, _, _,euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch,yaw,roll =map(lambda temp:temp[0],euler_angles)
    return pitch,yaw,roll

    # head_pose = [ rmat[0,0], rmat[0,1], rmat[0,2], tvec[0],

                   # rmat[1,0], rmat[1,1], rmat[1,2], tvec[1],

                   # rmat[2,0], rmat[2,1], rmat[2,2], tvec[2],

                         # 0.0,      0.0,        0.0,    1.0 ]

    #print(head_pose) #TODO remove this line

    # return self.rotationMatrixToEulerAngles(rmat)
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).

def rotationMatrixToEulerAngles(R) :
    #assert(isRotationMatrix(R))
    #To prevent the Gimbal Lock it is possible to use
    #a threshold of 1e-6 for discrimination
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])