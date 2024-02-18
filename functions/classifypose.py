import matplotlib.pyplot as plt
import cv2

def classifypose(output_image, display = False):
    label = "Unknown Pose"
    color = (0,255,0)
    Joints(KEYPOINT_DICT,keypoints_with_scores)
    left_elbow_angle = CalculateAngle(KEYPOINT_DICT['left_shoulder'],
                                     KEYPOINT_DICT['left_elbow'],
                                     KEYPOINT_DICT['left_wrist'])
    right_elbow_angle = CalculateAngle(KEYPOINT_DICT['right_shoulder'],
                                       KEYPOINT_DICT['right_elbow'],
                                       KEYPOINT_DICT['right_wrist'])
    right_shoulder_angle = CalculateAngle(KEYPOINT_DICT['right_elbow'],
                                       KEYPOINT_DICT['right_shoulder'],
                                       KEYPOINT_DICT['right_hip'])
    left_shoulder_angle = CalculateAngle(KEYPOINT_DICT['left_elbow'],
                                       KEYPOINT_DICT['left_shoulder'],
                                       KEYPOINT_DICT['left_hip'])
    left_knee_angle = CalculateAngle(KEYPOINT_DICT['left_hip'],
                                       KEYPOINT_DICT['left_knee'],
                                       KEYPOINT_DICT['left_ankle'])
    right_knee_angle = CalculateAngle(KEYPOINT_DICT['right_hip'],
                                       KEYPOINT_DICT['right_knee'],
                                       KEYPOINT_DICT['right_ankle'])    
    if left_elbow_angle > 90 and left_shoulder_angle > 90:
        label = 'left hand up'
    if right_elbow_angle > 90 and right_shoulder_angle > 90:
        label = 'right hand up'
    if (left_elbow_angle > 90 and left_shoulder_angle > 90) and (right_elbow_angle > 90 and right_shoulder_angle>90):
        label = 'both hands up'
    cv2.putText(output_image, label, (10,30), cv2.FONT_HERSHEY_PLAIN,2,color,2)
    if display:
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-10]);plt.title('Output image');pltt.axis('off');
    else:
        return output_image, label