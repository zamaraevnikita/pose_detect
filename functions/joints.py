def Joints(KEYPOINT_DICT,keypoints_with_scores):
    try:
        for i, j in zip(KEYPOINT_DICT, range(17)) :
            if keypoints_with_scores[0][0][j][2] > 0.4:
                KEYPOINT_DICT[i] = keypoints_with_scores[0][0][j]
    except:
        pass