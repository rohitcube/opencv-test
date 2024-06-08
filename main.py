import cv2
import math
import numpy as np
import mediapipe as mp

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle >180.0:
        angle = 360-angle

    return angle

#define coordinates


#implement a function that returns number of reps
def shoulderpress(lshoulder, rshoulder, lelbow, relbow, lwrist, rwrist):
        lshoulder_angle = calculate_angle(rshoulder, lshoulder, lelbow)
        lelbow_angle = calculate_angle(lshoulder, lelbow, lwrist)
        rshoulder_angle = calculate_angle(lshoulder, rshoulder, relbow)
        relbow_angle = calculate_angle(rshoulder, relbow, rwrist)
        return lshoulder_angle, lelbow_angle, rshoulder_angle, relbow_angle




def calculate_angle2(a, b):
    """
    Calculate the angle (in radians) between two sets of (x, y) coordinates.

    Args:
        x1 (float): The x-coordinate of the first point.
        y1 (float): The y-coordinate of the first point.
        x2 (float): The x-coordinate of the second point.
        y2 (float): The y-coordinate of the second point.

    Returns:
        float: The angle (in radians) between the two sets of coordinates.
    """
    a = np.array(a) # First
    b = np.array(b) # Mid
    # Calculate the differences between the x and y coordinates
    dx = a[0] - b[0]
    dy = a[1] - b[1]

    # Calculate the angle using the atan2 function
    angle = np.arctan2(dy, dx)

    return angle

def calculate_x_coord_diff_value(a,b):
    a = np.array(a) # First
    b = np.array(b) # Mid
    x_vector = int(a[0] - b[0])
    return x_vector

cap = cv2.VideoCapture(0)


elbow_is_right_angle = True
# The condition to evaluate whether a rep counts or not,
wrist_in_line_with_nose = False
#
elbow_crosses_shoulder_line = False
reps = 0


# assigning two modules from the MediaPipe library to variables mp_drawing and mp_pose.
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence=0.5) as pose:


    while cap.isOpened() == True:
    # returns numpy array that shows whether image is returned properly
    # ret = returns false or true, if camera is being used by something else
        ret, image = cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            lshoulder_angle, lelbow_angle, rshoulder_angle, relbow_angle = shoulderpress(lshoulder, rshoulder, lelbow, relbow, lwrist, rwrist)

            # Calculate angle
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            nose_wrist_angle = calculate_angle2(wrist, nose)
            elbow_wrist_x_coord_difference = abs(calculate_x_coord_diff_value(wrist, nose))

            # Visualize angle
            cv2.putText(image, str(elbow_angle),
                           tuple(np.multiply(elbow, [640, 480]).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )

            # nose wrist in line check
            if (nose_wrist_angle < 20 or nose_wrist_angle > -20):
              wrist_in_line_with_nose = True

            # elbow angle check
            if (elbow_wrist_x_coord_difference < 0.8):
                elbow_is_right_angle = False

            # this should reset aither every upward movement or every downward movement
            if elbow_is_right_angle:
                text = "GOOD FORM!" # Green color if boolean_value is True
            else:
                 text = "ELBOW AINT RIGHT"

            x, y = 50, 250

            lower_threshold, upper_threshold = 80, 100 #can be adjusted

            if lshoulder_angle - lelbow_angle < lower_threshold:
                text2 = "left forearm out wide"
            elif lshoulder_angle - lelbow_angle > upper_threshold:
                text2 = "left forearm leaned in"
            elif rshoulder_angle - relbow_angle < lower_threshold:
                text2 = "right forearm out wide"
            elif rshoulder_angle - relbow_angle > upper_threshold:
                text2 = "right forearm leaned in"
            else:
                text2 = "Good Form!"

            # 16, 10 on a straight line, nose and wrist are in the same line

            cv2.putText(image, text2, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(elbow_crosses_shoulder_line), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(reps), (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            if wrist_in_line_with_nose:
                if elbow_crosses_shoulder_line and elbow_is_right_angle:
                    reps += 1
                # Reset all conditions
                elbow_is_right_angle = True
                wrist_in_line_with_nose = False
                elbow_crosses_shoulder_line = False


        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

        k = cv2.waitKey(1)

        """
        if k == 117:
            export_landmark(results, 'up')
        if k == 100:
            export_landmark(results, 'down')
        """

        cv2.imshow("Raw feed", image)
        if cv2.waitKey(1) == ord('q'):
            # if key equals to ordinal value of 'q' loop breaks
            break


print('done')
cap.release()
cv2.destroyAllWindows()

