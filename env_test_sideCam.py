import cv2
import mediapipe as mp
import math
import logging as logg
import time
import requests
from collections import deque

# CHỈ MỞ SIDE CAM (Kiểm tra xem camera sườn của bạn là index 0 hay 1, ở đây tôi để mặc định là 0 để tiện test đơn lập)
side_cam = cv2.VideoCapture(1) 

# Khởi tạo Mediapipe-pose (Chỉ giữ lại camera bên)
mpPose = mp.solutions.pose 
pose_side = mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils 

last_alert = 0
alert_sent = False

torso_history = deque(maxlen=15)
neck_history = deque(maxlen=15)

last_sent_status = None
status_start_time = None 
bad_start_time = None 
current_detected_status = None
STATUS_DELAY = 5
WARNING_DELAY = 6
BAD_DELAY = 7
ALERT_DELAY = 15
token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0aGFuaG5ndXllbnNvbmpxa0BnbWFpbC5jb20iLCJyb2xlIjoiVVNFUiIsImlhdCI6MTc4MjU1ODA3MywiZXhwIjoxNzgyNTU4OTczfQ.Gu-uMOYEvznNG_oR1FRBGN5vcBuYVsrIhQLrollholY"
SESSION_ID = 4

## Kiểm tra thân người (Góc nghiêng lưng)
def torso_angle(res):
    if not res.pose_landmarks:
        return None
    lm = res.pose_landmarks.landmark
    # Dùng điểm số 12 (Vai phải) và 24 (Hông phải)
    shoulder_x, shoulder_y = lm[12].x, lm[12].y
    hip_x, hip_y = lm[24].x, lm[24].y
    
    delta_x, delta_y = shoulder_x - hip_x, shoulder_y - hip_y

    # Góc thân so với phương thẳng đứng
    angle = math.degrees(math.atan2(delta_x, -delta_y))
    torso_history.append(angle)
    return sum(torso_history) / len(torso_history)

# Kiểm tra cúi cổ
def neck_angle(res, current_torso_ang):
    if not res.pose_landmarks:
        return None

    lm = res.pose_landmarks.landmark

    if (lm[8].visibility < 0.4 or
        lm[12].visibility < 0.5 or
        lm[24].visibility < 0.5):
        return None
   
    ear = lm[8]         # Tai phải
    shoulder = lm[12]   # Vai phải
    
    # Nếu torso_angle tính góc từ hàm trên mượt hơn, ta dùng trực tiếp nó thay vì tính lại torso_angle gốc ở đây
    # Điều này giúp đồng bộ góc giữa 2 hàm tốt hơn
    torso_ang_val = current_torso_ang if current_torso_ang is not None else 0.0

    # Góc của đầu/cổ tạo bởi tai và vai
    head_dx = ear.x - shoulder.x
    head_dy = shoulder.y - ear.y
    head_angle = math.degrees(math.atan2(head_dx, head_dy))

    # Góc cổ thực tế = Góc đầu - Góc thân
    neck = abs(head_angle - torso_ang_val)
    neck_history.append(neck)

    return sum(neck_history) / len(neck_history)

def send_snapshot(token, session_id, shouder_r, torso_a, neck_a, status):
    try:
        http_res = requests.post("http://localhost:8080/api/posture-snapshots/create",
                            headers={
                                "Content-Type": "application/json",
                                "Authorization": f"Bearer {token}"
                            },
                            json={
                                "sessionId": session_id,
                                "shoulderRatio": shouder_r,
                                "torsoAngle": torso_a,
                                "neckAngle": neck_a,
                                "postureStatus": status
                            }
                        )
        print(f"Snapshot sent [{status}]: {http_res.status_code}")
    except Exception as e:
        logg.error(f"Failed to send snapshot: {e}")

def send_message(token, status):
    try:
        http_res = requests.post("http://localhost:8080/api/mqtt/alert",
                            headers={
                                "Content-Type": "application/json",
                                "Authorization": f"Bearer {token}"
                            },
                            json={
                                "status": status,
                            }      
                        )
        print(f"Message sent [{status}]: {http_res.status_code}")
    except Exception as e:
        logg.error(f"Failed to send message: {e}")

while True:
    ret_side, frame_side = side_cam.read()
    
    # Giả lập hoặc gán mặc định trạng thái vai trước (Front Cam) là luôn đúng
    shoulder_state = "STRAIGHT"
    torso_state, neck_state = None, None
    level = None
    shouder_ang, torso_ang, neck_ang = 0.0, None, None

    if ret_side:
        size_rgb = cv2.cvtColor(frame_side, cv2.COLOR_BGR2RGB)
        res_side = pose_side.process(size_rgb)
        
        if res_side.pose_landmarks:
            mpDraw.draw_landmarks(frame_side, res_side.pose_landmarks, mpPose.POSE_CONNECTIONS)
            torso_ang = torso_angle(res_side)
            neck_ang = neck_angle(res_side, torso_ang)

    # Đánh giá trạng thái lưng (Torso)
    if torso_ang is not None:
        if torso_ang <= 10:
            torso_state = "STRAIGHT"
        elif torso_ang <= 20:
            torso_state = "WARNING"
        else:
            torso_state = "BAD"

    # Đánh giá trạng thái cổ (Neck)
    if neck_ang is not None:
        # Nếu đang ngồi dựa lưng ra sau (góc âm) thì ngưỡng cổ khác
        if torso_ang is not None and torso_ang < -10:
            if neck_ang <= 25:
                neck_state = "STRAIGHT"
            elif neck_ang <= 35:
                neck_state = "WARNING"
            else:
                neck_state = "BAD"
        # Ngồi bình thường / Gù về trước
        else:
            if neck_ang <= 18:
                neck_state = "STRAIGHT"
            elif neck_ang <= 30:
                neck_state = "WARNING"
            else:
                neck_state = "BAD"

    warning_count = 0
    bad_count = 0
    messages = []

    ######### Kiểm tra Lưng (Torso) ##############
    if torso_state == "WARNING":
        warning_count += 1
        messages.append("GU LUNG")
    elif torso_state == "BAD":
        bad_count += 1
        messages.append("GU LUNG")

    ######### Kiểm tra Cổ (Neck) ##############
    if neck_state == "WARNING":
        warning_count += 1
        messages.append("GU CO")
    elif neck_state == "BAD":
        bad_count += 1
        messages.append("GU CO")
    
    ######### Kết quả tổng hợp ##############
    if torso_state is None or neck_state is None:
        level = "NO_PERSON"
    elif bad_count > 0:
        level = "BAD_POSTURE"
    elif warning_count > 0:
        level = "WARNING_POSTURE"
    else:
        level = "GOOD_POSTURE"

    if level == "GOOD_POSTURE":
        final_text = "TU THE DUNG"
    elif level == "NO_PERSON":
        final_text = "KHONG CO NGUOI"
    else:
        final_text = " + ".join(messages)
    
    # Màu sắc hiển thị
    if level == "GOOD_POSTURE":
        final_color = (0, 255, 0)
    elif level == "WARNING_POSTURE":
        final_color = (0, 255, 255)
    elif level == "BAD_POSTURE":
        final_color = (0, 0, 255)
    else:
        final_color = (123, 234, 111)

    # Hiển thị thông số lên màn hình Side Cam
    if ret_side:
        cv2.putText(
            frame_side,
            f"{level}: {final_text}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            final_color,
            2)

        if torso_ang is not None:
            cv2.putText(
                frame_side,
                f"Torso: {torso_ang:.1f}",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                final_color,
                2)

        if neck_ang is not None:
            cv2.putText(
                frame_side,
                f"Neck: {neck_ang:.1f}",
                (30, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                final_color,
                2)

    current_status = (level, final_text)
    
    ############# Xử lý gửi API trì hoãn #############
    if level == "GOOD_POSTURE":
        status_start_time = None  
        if current_status != last_sent_status:
            send_snapshot(token, SESSION_ID, shouder_ang, torso_ang, neck_ang, level)
            send_message(token, level)
            last_sent_status = current_status
        bad_start_time = None
        alert_sent = False
    elif level != "NO_PERSON": 
        if current_status != current_detected_status:
            current_detected_status = current_status
            status_start_time = time.time()
        if status_start_time is None:
            status_start_time = time.time()
       
        posture_duration = time.time() - status_start_time
        delay = WARNING_DELAY if level == "WARNING_POSTURE" else BAD_DELAY

        if posture_duration >= delay:
            if current_status != last_sent_status:
                send_snapshot(token, SESSION_ID, shouder_ang, torso_ang, neck_ang, level)
                send_message(token, level)
                last_sent_status = current_status

    # CHỈ HIỂN THỊ CỬA SỔ SIDE CAM
    if ret_side: 
        cv2.imshow("Side Cam - Torso & Neck", frame_side)

    if cv2.waitKey(1) == ord('q'):
        break

side_cam.release()
cv2.destroyAllWindows()