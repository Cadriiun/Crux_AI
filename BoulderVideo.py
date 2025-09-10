from turtle import color, right
import cv2
from scipy import linalg
from ultralytics import YOLO
import numpy as np
from collections import deque

# this is good for data capturing 
class BoulderingSimulation:
    KEYPOINTS = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]
    CLIMBINGPOSITION = [
        (5, 6), (5, 7), (7, 9),    # Left arm
        (6, 8), (8, 10),            # Right arm
        (11, 13), (13, 15),         # Left leg
        (12, 14), (14, 16),         # Right leg
        (5, 11), (6, 12)    
    ]

    def __init__(self):
        # get the model and load it

        self.hold_model = YOLO(
            r'runs\detect\boulder_detection7\weights\best.pt')
        self.pose_model = YOLO('yolo11n-pose.pt').cuda()

        self.grab_distance_threshold = 40
        self.balance_factor_threshold = 0.7
        self.extension_threshold = 0.8
        self.history_length = 10
        self.pose_history = deque(maxlen=self.history_length)

        self.holds_color = {
            'grabbed': (0, 255, 255),
            'ungrabbed': (255, 0, 0),
        }
        self.connection_colors = {
            'normal': (0, 255, 0),
            'extended': (0, 165, 255),
            'flexed': (255, 0, 0)
        }
    # detect holds by frame and classify them

    def detected_holds(self, frame):
        results = self.hold_model(frame)

        holds = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            holds.append({
                'box': (x1, y1, x2, y2),
                'center': ((x1+y1)//2, (x2+y2)//2),
                'grabbed': False,
                'difficulty': 'easy'
            })
        return holds

    # analyze on how to climb by frame and interact with the holds
    def analyze_frame(self, frame):
        # detect holds
        holds = self.detected_holds(frame)
        
        # Estimate pose
        pose_results = self.pose_model(frame)
        keypoints = pose_results[0].keypoints.xy[0].cpu().numpy()

        self.pose_history.append(keypoints)
        analysis = {
            'holds': holds,
            'keypoints': keypoints,
            'grab_analysis': self._analyze_hold_interaction(keypoints, holds),
            'limb_analysis': self._analyze_limb_usage(keypoints),
            'balance': self._calculate_climbing_balance(keypoints),
            'technique': self._identify_climbing_technique(),
        }
        return analysis

    # check if the holds are being grabbed and the hands positions
    def _analyze_hold_interaction(self, keypoints, holds):
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]

        grab_analysis = {
            'left_hand': {'holdsIndex': None, 'distance': float('inf')},
            'right_hand': {'holdsIndex': None, 'distance': float('inf')},
            'grabbed_holds': []
        }

        for i, hold in enumerate(holds):
            # getting the distance of left and right wrist
            left_distance = np.linalg.norm(left_wrist - hold['center'])
            right_distance = np.linalg.norm(right_wrist - hold['center'])

            # update the grab
            if left_distance < grab_analysis['left_hand']['distance']:
                grab_analysis['left_hand'] = {
                    'holdsIndex': i, 'distance': left_distance}
            if right_distance < grab_analysis['right_hand']['distance']:
                grab_analysis['right_hand'] = {
                    'holdsIndex': i, 'distance': right_distance}

            # update whether it grabbed or not
            if min(left_distance, right_distance) < self.grab_distance_threshold:
                hold['grabbed'] = True
            else:
                hold['grabbed'] = False

        return grab_analysis

    # analyze limb angles
    def _analyze_limb_usage(self, keypoints):
        limb_analysis = {}

        # check left and right
        for side in ['left', 'right']:
            # getting the value from keypoints
            shoulder = keypoints[5 if side == 'left' else 6]
            elbow = keypoints[7 if side == 'left' else 8]
            wrist = keypoints[9 if side == 'right' else 10]

            # get the extension ratio
            upper_distance = np.linalg.norm(elbow-shoulder)
            lower_distance = np.linalg.norm(wrist - elbow)
            total_distance = upper_distance + lower_distance
            extension_ratio = total_distance / \
                (np.linalg.norm(wrist-shoulder)
                 if np.linalg.norm(wrist-shoulder) > 0 else 1)

            # add it to limb analysis
            limb_analysis[f'{side}_arm'] = {
                'extension': extension_ratio,
                'status': 'extended' if extension_ratio > self.extension_threshold else 'flexed'
            }

        return limb_analysis

    # calculate normal balance for bouldering
    def _calculate_climbing_balance(self, keypoints):
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]

        # calculate the center of mass and check for balance
        COM_X = (left_shoulder[0] + right_shoulder[0] +
                 left_hip[0] + right_hip[0])/4
        COM_Y = (left_shoulder[1] + right_shoulder[1] +
                 left_hip[1] + right_hip[1])/4

        support_keypoints = [
            keypoints[9],  # left wrist
            keypoints[10],  # right wrist
            keypoints[14],  # left ankle
            keypoints[15],  # right ankle
        ]

        return self._com_balance_matric((COM_X, COM_Y), support_keypoints)

    # calculate the center of mass(got it from chatgpt)
    def _com_balance_matric(self, COM, supportKeyPoints):
        hull = cv2.convexHull(np.array(supportKeyPoints, dtype=np.float32))
        return cv2.pointPolygonTest(hull, (float(COM[0]), float(COM[1])), False) >= 0

    # get the technique from pose history (Static dynamic or Controlled)
    def _identify_climbing_technique(self):
        if len(self.pose_history) < 5:
            return "Initializing... "

        hand_movement = []
        for i in range(1, len(self.pose_history)):
            move = np.linalg.norm(self.pose_history[i][9] - self.pose_history[i-1][9]) + np.linalg.norm(
                self.pose_history[i][10] - self.pose_history[i][10])
            hand_movement.append(move)

        avg_move = np.mean(hand_movement)

        if avg_move < 5:
            return "Static"
        elif any(m > 50 for m in hand_movement):
            return "Dynamic"
        else:
            return "Controlled"

    # draw the figure
    def visualize_analysis(self, frame, analysis):

        # draw rectangular box and lables
        for i, hold in enumerate(analysis['holds']):
            color = self.holds_color['grabbed'] if hold['grabbed'] else self.holds_color['ungrabbed']
            cv2.rectangle(frame, hold['box'][:2], hold['box'][2:], color, 2)
            cv2.putText(frame, str(
                i), (hold['box'][0], hold['box'][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # just drawing the body keypoints
        for i, kp in enumerate(analysis['keypoints']):
            if i < 18:
                color = (0, 255, 255) if i in [
                    10, 11, 16, 17] else (255, 255, 255)
                cv2.circle(frame, tuple(kp.astype(int)), 5, color, -1)

        # draw limbs
        for i, j in self.CLIMBINGPOSITION:
            start = tuple(analysis['keypoints'][i].astype(int))
            end = tuple(analysis['keypoints'][j].astype(int))

            # determine limb status
            if {i, j}.issubset({5, 7, 9}):  # this will be left arm
                status = analysis['limb_analysis']['left_arm']['status']
            elif {i, j}.issubset({6, 8, 10}):  # this will be right arm
                status = analysis['limb_analysis']['right_arm']['status']
            else:
                status = 'normal'

            cv2.line(frame, start, end, self.connection_colors[status], 2)
        # Just displaying text
        cv2.putText(frame, f"Technique: {analysis['technique']}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Balance: {analysis['balance']:.2f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Grabbed Holds: {len(analysis['grab_analysis']['grabbed_holds'])}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame


if __name__ == "__main__":
    analyzer = BoulderingSimulation()
    

    cap = cv2.VideoCapture("input_video.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        analysis = analyzer.analyze_frame(frame)
        visualized = analyzer.visualize_analysis(frame.copy(), analysis)
        
        cv2.imshow("Bouldering Analysis", visualized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()