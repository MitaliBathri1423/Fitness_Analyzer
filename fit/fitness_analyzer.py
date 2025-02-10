import cv2 as cv
import numpy as np
import argparse
import math

# ... (BODY_PARTS and POSE_PAIRS remain the same)
BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

def estimate_pose(frame, net, inWidth, inHeight, args):
    """Estimates the pose from a frame using OpenPose."""
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponding body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > args.thr else None)
    return points


def draw_pose(frame, points):
    """Draws the pose skeleton on the frame, including angle lines."""
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    # Draw angle lines (right side)
    if points[BODY_PARTS["RHip"]] and points[BODY_PARTS["RKnee"]] and points[BODY_PARTS["RAnkle"]]:
        cv.line(frame, points[BODY_PARTS["RHip"]], points[BODY_PARTS["RKnee"]], (255, 0, 0), 2)  # Blue line
        cv.line(frame, points[BODY_PARTS["RKnee"]], points[BODY_PARTS["RAnkle"]], (255, 0, 0), 2)  # Blue line

    # Draw angle lines (left side)
    if points[BODY_PARTS["LHip"]] and points[BODY_PARTS["LKnee"]] and points[BODY_PARTS["LAnkle"]]:
        cv.line(frame, points[BODY_PARTS["LHip"]], points[BODY_PARTS["LKnee"]], (0, 0, 255), 2)  # Red line
        cv.line(frame, points[BODY_PARTS["LKnee"]], points[BODY_PARTS["LAnkle"]], (0, 0, 255), 2)  # Red line


def calculate_angle(p1, p2, p3):
    """Calculates the angle between three points."""
    if all(p is not None for p in [p1, p2, p3]):  # Check if all points are detected or not
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)

        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm == 0 or v2_norm == 0:
            return 0  # Handle cases where a vector has zero length

        angle = math.degrees(np.arccos(np.dot(v1, v2) / (v1_norm * v2_norm)))
        return angle
    else:
        return None  # Return None if any point is None
    
def analyze_squat_form(points):
    right_hip = points[BODY_PARTS["RHip"]]
    right_knee = points[BODY_PARTS["RKnee"]]
    right_ankle = points[BODY_PARTS["RAnkle"]]
    right_shoulder = points[BODY_PARTS["RShoulder"]]

    left_hip = points[BODY_PARTS["LHip"]]
    left_knee = points[BODY_PARTS["LKnee"]]
    left_ankle = points[BODY_PARTS["LAnkle"]]
    left_shoulder = points[BODY_PARTS["LShoulder"]]

    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

    feedback_lines = []

    if right_knee_angle is not None and left_knee_angle is not None:
        avg_knee_angle = (right_knee_angle + left_knee_angle) / 2
        if avg_knee_angle < 80:
            feedback_lines.append("Squat Depth: Go Deeper!")
        elif avg_knee_angle > 115:
            feedback_lines.append("Squat Depth: Don't go too deep.")
        else:
            feedback_lines.append("Squat Depth: Good Depth!")

        # Check for knee valgus/varus (simplified)
        if right_hip and right_knee and right_ankle and left_hip and left_knee and left_ankle:
            right_hip_knee_dist = np.linalg.norm(np.array(right_hip) - np.array(right_knee))
            right_knee_ankle_dist = np.linalg.norm(np.array(right_knee) - np.array(right_ankle))
            left_hip_knee_dist = np.linalg.norm(np.array(left_hip) - np.array(left_knee))
            left_knee_ankle_dist = np.linalg.norm(np.array(left_knee) - np.array(left_ankle))

            if right_knee_ankle_dist > right_hip_knee_dist * 1.2 or left_knee_ankle_dist > left_hip_knee_dist * 1.2:
                feedback_lines.append("Knee Alignment: Knees are caving in!")
            elif right_knee_ankle_dist * 1.2 < right_hip_knee_dist or left_knee_ankle_dist * 1.2 < left_hip_knee_dist:
                feedback_lines.append("Knee Alignment: Knees are going too far out!")
            else:
                feedback_lines.append("Knee Alignment: Good Knee Alignment!")

        # Back angle (simplified)
        if right_shoulder and right_hip and right_knee and left_shoulder and left_hip and left_knee:
            right_back_angle = calculate_angle(right_shoulder, right_hip, right_knee)
            left_back_angle = calculate_angle(left_shoulder, left_hip, left_knee)
            avg_back_angle = (right_back_angle + left_back_angle) / 2
            if avg_back_angle is not None:
                if avg_back_angle < 150:
                    feedback_lines.append("Back Posture: Keep your back straighter!")
                else:
                    feedback_lines.append("Back Posture: Good back posture!")

    return feedback_lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', help='Path to image or video. Skip to capture frames from camera')
    parser.add_argument('--thr', default=0.3, type=float, help='Threshold value for pose parts heat map')
    parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
    parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

    args = parser.parse_args()

    # cap = cv.VideoCapture(args.input if args.input else 0)

    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-input', help='Path to image or video')
        args = parser.parse_args()

    if args.input:  # Check if -input was provided
        print(f"Using input: {args.input}")
        cap = cv.VideoCapture(args.input)  # Open video or image
    else:
        print("Using default camera (webcam)")
        cap = cv.VideoCapture(0)  # Open webcam
    net = cv.dnn.readNetFromTensorflow(r"C:\Users\mital\Desktop\major-project\FitnessApplication\graph_opt.pb")  # Make sure the path is correct
    inWidth = args.width
    inHeight = args.height

    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))  # Get frame width
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) # Get frame height

    cv.namedWindow('Fitness Form Analyzer', cv.WINDOW_NORMAL)  # Create a resizable window first

    # Optional: Resize the window to match the video/webcam resolution before going fullscreen.
    cv.resizeWindow('Fitness Form Analyzer', frameWidth, frameHeight)

    cv.setWindowProperty('Fitness Form Analyzer', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN) # Go fullscreen

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        points = estimate_pose(frame, net, inWidth, inHeight, args)
        draw_pose(frame, points)

        feedback_lines = analyze_squat_form(points)

        y_offset = 50
        for line in feedback_lines:
            cv.putText(frame, line, (10, y_offset), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))
            y_offset += 30
        t, _ = net.getPerfProfile()
        freq = cv.getTickFrequency() / 1000
        cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        cv.imshow('Fitness Form Analyzer', frame)

    cap.release()
    cv.destroyAllWindows()  # Important: Close all windows

if __name__ == "__main__":
    main()

