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

def calculate_angle_with_negative(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    return angle

def calculate_angle2(a,b):
    a = np.array(a) # First
    b = np.array(b) # Mid
    radians = np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return angle

#define coordinates


#implement a function that returns number of reps
def shoulderpress(lshoulder, rshoulder, lelbow, relbow, lwrist, rwrist):
        lshoulder_angle = calculate_angle(rshoulder, lshoulder, lelbow)
        lelbow_angle = calculate_angle(lshoulder, lelbow, lwrist)
        rshoulder_angle = calculate_angle(lshoulder, rshoulder, relbow)
        relbow_angle = calculate_angle(rshoulder, relbow, rwrist)
        return lshoulder_angle, lelbow_angle, rshoulder_angle, relbow_angle


def calculate_x_coord_diff_value(a,b):
    a = np.array(a) # First
    b = np.array(b) # Mid
    x_vector = int(a[0] - b[0])
    return x_vector

cap = cv2.VideoCapture(0)


elbow_is_right_angle = True
# The condition to evaluate whether a rep counts or not, both have to be true, and elbow is right angle has to be true
wrist_in_line_with_nose = False
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
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            l_index_finger = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
            r_index_finger = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]
            lshoulder_angle, lelbow_angle, rshoulder_angle, relbow_angle = shoulderpress(lshoulder, rshoulder, lelbow, relbow, lwrist, rwrist)
            lmouth = [landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x,landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y]
            rmouth = [landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x,landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y]
            # Calculate angle
            relbow_angle = calculate_angle(rshoulder, relbow, wrist)
            elbow_shoulder_angle = calculate_angle_with_negative(rshoulder, lshoulder, relbow)
            lshoulder_rshoulder_mouth = calculate_angle(lshoulder, rshoulder, rmouth)
            rshoulder_rmouth_rightindex = calculate_angle(rshoulder, rmouth, r_index_finger)

            # Visualize angle
            '''
            cv2.putText(image, str(relbow_angle),
                           tuple(np.multiply(elbow, [640, 480]).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            '''
            # nose wrist in line check
            if (abs(lshoulder_rshoulder_mouth-rshoulder_rmouth_rightindex) < 9):
                mouth_in_line_with_finger = True
            else:
                mouth_in_line_with_finger = False

            # elbow shoulder in line check


            # this should reset aither every upward movement or every downward movement
            if mouth_in_line_with_finger:
                loop = True  # Green color if boolean_value is True
            else:
                loop = False #"NOT STARTING POS"

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

            cv2.putText(image, text2, (x, y), cv2.FONT_HERSHEY_TRIPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(elbow_shoulder_angle), (500,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
       #    cv2.putText(image, str(lshoulder_rshoulder_mouth), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        #   cv2.putText(image, str(rshoulder_rmouth_rightindex), (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # check rep
            if wrist_in_line_with_nose:
                if mouth_in_line_with_finger:
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

