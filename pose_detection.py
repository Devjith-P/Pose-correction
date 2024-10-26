import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose and Drawing Utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Placeholder for dynamic pose criteria
POSES = {
    'front_double_biceps': {},
    'side_chest': {},
    'front_lat_spread': {}
}

current_pose = 'front_double_biceps'
capture_reference_pose = False

def calculate_angle(point1, point2, point3):
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        # Resize and convert to RGB
        image = cv2.resize(frame, (640, 480))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Process the image and get pose landmarks
        results = pose.process(image_rgb)

        # Convert back to BGR and make image writeable for drawing
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Draw key points and provide feedback based on the selected pose
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Get landmark positions
            landmarks = results.pose_landmarks.landmark
            
            # Arm and chest positions
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            # Calculate angles
            left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
            chest_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y

            if capture_reference_pose:
                # Set reference criteria for the current pose
                POSES[current_pose] = {
                    'left_arm_angle': left_arm_angle,
                    'right_arm_angle': right_arm_angle,
                    'left_leg_angle': left_leg_angle,
                    'right_leg_angle': right_leg_angle,
                    'chest_level': chest_y
                }
                print(f"Reference angles captured for {current_pose}.")
                capture_reference_pose = False

            # Fetch the reference angles for the current pose
            pose_criteria = POSES[current_pose]
            if pose_criteria:
                target_left_arm_angle = pose_criteria['left_arm_angle']
                target_right_arm_angle = pose_criteria['right_arm_angle']
                target_left_leg_angle = pose_criteria['left_leg_angle']
                target_right_leg_angle = pose_criteria['right_leg_angle']
                target_chest_level = pose_criteria['chest_level']

                # Feedback for Left Arm
                if abs(left_arm_angle - target_left_arm_angle) > 10:
                    adjustment = "Raise" if left_arm_angle < target_left_arm_angle else "Lower"
                    cv2.putText(image, f"{adjustment} Left Elbow", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, "Left Arm Good", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                # Feedback for Right Arm
                if abs(right_arm_angle - target_right_arm_angle) > 10:
                    adjustment = "Raise" if right_arm_angle < target_right_arm_angle else "Lower"
                    cv2.putText(image, f"{adjustment} Right Elbow", (400, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, "Right Arm Good", (400, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                # Feedback for Chest Level - inverted logic
                if chest_y < target_chest_level - 0.05:
                    cv2.putText(image, "Raise Chest", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                elif chest_y > target_chest_level + 0.05:
                    cv2.putText(image, "Lower Chest", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, "Chest Good", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                # Feedback for Left Leg Position
                if abs(left_leg_angle - target_left_leg_angle) > 10:
                    leg_adjust = "Straighten" if left_leg_angle < target_left_leg_angle else "Bend"
                    cv2.putText(image, f"{leg_adjust} Left Leg", (50, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, "Left Leg Good", (50, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                # Feedback for Right Leg Position
                if abs(right_leg_angle - target_right_leg_angle) > 10:
                    leg_adjust = "Straighten" if right_leg_angle < target_right_leg_angle else "Bend"
                    cv2.putText(image, f"{leg_adjust} Right Leg", (50, 200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, "Right Leg Good", (50, 200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the image
        cv2.imshow('Pose Detection', image)

        # Keyboard controls for different poses and capturing reference poses
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            capture_reference_pose = True
        elif key == ord('1'):
            current_pose = 'front_double_biceps'
        elif key == ord('2'):
            current_pose = 'side_chest'
        elif key == ord('3'):
            current_pose = 'front_lat_spread'

cap.release()
cv2.destroyAllWindows()
