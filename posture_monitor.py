import cv2
import mediapipe as mp
import math
import logging as logg
import time
import requests
from collections import deque

front_cam = cv2.VideoCapture(0)
side_cam = cv2.VideoCapture(1)

#Initalize Mediapipe-pose
mpPose = mp.solutions.pose #lấy module Pose từ MediaPipe
pose_front = mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose_side = mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils #Khởi tạo lớp vẽ của Mediapipe

last_alert = 0
alert_sent = False


list = []
shouder_history = deque(maxlen = 15) #Luu toi da 15 phan tu
torso_history = deque(maxlen = 15)
neck_history = deque(maxlen = 15)
upper_history = deque(maxlen=15)

last_sent_status = None
status_start_time = None # Timer cho STATUS_DELAY (gửi snapshot)
bad_start_time = None # Timer riêng cho ALERT_DELAY (beep/alert)
current_detected_status = None
STATUS_DELAY = 5
WARNING_DELAY = 10
BAD_DELAY = 15 
ALERT_DELAY = 15


token  = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0aGFuaG5ndXllbnNvbmpxa0BnbWFpbC5jb20iLCJyb2xlIjoiVVNFUiIsImlhdCI6MTc4Mjc5NjczMywiZXhwIjoxNzgyNzk3NjMzfQ.A3PyOeA-i-EF5cFMvxO9fnZtn4CfYga3Azb27KRzwHQ"
SESSION_ID = 4
    



def make_lm_timestaps(res): #Chứa toạ độ các điểm trên khung xương
    lm_list = []
    id = 0
    for lm in res.pose_landmarks.landmark:
        lm_list.append(f"idx:{id},x:{lm.x},y:{lm.y},z:{lm.z}")
        id+=1
    return lm_list 

def is_visibility(lm1,lm2):
    return lm1.visibility > 0.5 and lm2.visibility > 0.5


#Kiem tra vai
def compare_Diff_Shoulder(res,frame):
    if not res.pose_landmarks:
        logg.error("Không tìm thấy người")
        return None;
    lm = res.pose_landmarks.landmark
    h,w,_ = frame.shape

    if not is_visibility(lm[11],lm[12]):
        return None

    left_shoulder_x,left_shoulder_y = int(lm[11].x * w),int(lm[11].y * h) #Toa do vai trai tren man anh
    right_shoulder_x,right_shoulder_y = int(lm[12].x * w),int(lm[12].y * h) #Toa do vai phai tren man anh

    diff_y = abs(left_shoulder_y- right_shoulder_y)
    diff_x = abs(left_shoulder_x - right_shoulder_x)

    if diff_x == 0: return None

    angle = math.degrees(math.atan2(abs(diff_y),abs(diff_x)))

    shouder_history.append(angle)
    smooth_ratio = sum(shouder_history) / len(shouder_history)
    return smooth_ratio


## Kiểm tra thân người
def torso_angle(res):
    if not res.pose_landmarks:
        logg.error("Không tìm thấy người")
        return None
    lm = res.pose_landmarks.landmark
    shoulder_x,shoulder_y = lm[12].x,lm[12].y
    hip_x,hip_y = lm[24].x,lm[24].y
    
    delta_x,delta_y = shoulder_x - hip_x,shoulder_y - hip_y

    # Góc thân so với trục Oy
    angle = math.degrees(math.atan2(delta_x,-delta_y))
    torso_history.append(angle)
    return sum(torso_history) /  len(torso_history)

# Kiểm tra cúi cổ
def neck_angle(res):
    if not res.pose_landmarks:
        logg.error("Không tìm thấy người")
        return None

    lm = res.pose_landmarks.landmark

    if(lm[8].visibility < 0.4 or lm[12].visibility < 0.5 or lm[24].visibility < 0.5):
        return None
   
    ear = lm[8]
    shoulder = lm[12]
    hip = lm[24]

    #Góc của thân người (tính vector từ hông -> vai)
    torso_dx = shoulder.x - hip.x
    torso_dy = shoulder.y - hip.y
    torso_angle = math.degrees(math.atan2(torso_dx, -torso_dy))

    #Góc của đầu (tính vector từ vai -> tai)
    head_dx = ear.x - shoulder.x
    head_dy = shoulder.y - ear.y
    head_angle = math.degrees(math.atan2(head_dx, head_dy))

    #Góc cổ = đầu - thân
    neck = abs(head_angle - torso_angle)

    neck_history.append(neck)

    return sum(neck_history) / len(neck_history)
    
def send_snapshot(token,session_id,shouder_r,torso_a,neck_a,status):
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

def send_message(token,status):
    try:
        http_res = requests.post("http://localhost:8080/api/mqtt/alert",
                            headers={
                                "Content-Type": "application/json",
                                "Authorization": f"Bearer {token}"
                            },
                            json={
                                "status":status,
                            }      
                        )
        print(f"Message sent [{status}]: {http_res.status_code}")
    except Exception as e:
        logg.error(f"Failed to send message: {e}")



def send_alert(token,session_id,status):
    try:
        http_res = requests.post("http://localhost:8080/api/alert/create",
                                headers={
                                    "Content-Type": "application/json",
                                    "Authorization": f"Bearer {token}"
                                },
                                json={
                                    "sessionId": session_id,
                                    "postureStatus": status,
                                    "message": "Bạn ngồi sai tư thế liên tục quá 30s"
                                }
                    )
        print(f"Alert sent: {http_res.status_code}")
    except Exception as e:
        logg.error(f"Failed to send alert: {e}")


while True:
    ret_front, frame_front = front_cam.read()
    ret_side, frame_side = side_cam.read()
    shoulder_state,torso_state,neck_state = None,None,None
    level = None
    shouder_ang,torso_ang,neck_ang= None,None,None

    if ret_front:
        frame_front = cv2.flip(frame_front,1)
        front_rgb = cv2.cvtColor(frame_front,cv2.COLOR_BGR2RGB)    
        res_front = pose_front.process(front_rgb) #Đưa ảnh vào cho AI phân tích

        if res_front.pose_landmarks:
            mpDraw.draw_landmarks(frame_front, res_front.pose_landmarks, mpPose.POSE_CONNECTIONS)
            #Kiểm tra ngồi lệch vai
            shouder_ang = compare_Diff_Shoulder(res_front,frame_front)


    if ret_side:
        size_rgb = cv2.cvtColor(frame_side,cv2.COLOR_BGR2RGB)
        res_side = pose_side.process(size_rgb)
        
        if res_side.pose_landmarks:
            mpDraw.draw_landmarks(frame_side, res_side.pose_landmarks, mpPose.POSE_CONNECTIONS)
            torso_ang = torso_angle(res_side)
            neck_ang = neck_angle(res_side)


    if shouder_ang is not None:
        if shouder_ang < 8:
            shoulder_state = "STRAIGHT"
        elif shouder_ang < 14:
            shoulder_state = "WARNING"
        else:
            shoulder_state = "BAD"


    if torso_ang is not None:
        if torso_ang <= 10:
            torso_state = "STRAIGHT"
        elif torso_ang <= 20:
            torso_state = "WARNING"
        else:
            torso_state = "BAD"

    if neck_ang is not None:

        # Nếu đang ngồi dựa lưng thì mới ngưỡng cổ
        if torso_ang is not None and torso_ang < -10:
            if neck_ang <= 30:
                neck_state = "STRAIGHT"
            elif neck_ang <= 38:
                neck_state = "WARNING"
            else:
                neck_state = "BAD"

        # Ngồi bình thường
        else:
            if neck_ang <= 20:
                neck_state = "STRAIGHT"
            elif neck_ang <= 35:
                neck_state = "WARNING"
            else:
                neck_state = "BAD"

    
    
    messages,warning_messages,bad_messages = [],[],[]

   #########Vai##############
    if shoulder_state == "WARNING":
        warning_messages.append("BAN DANG LECH VAI")

    elif shoulder_state == "BAD":
        bad_messages.append("CANH BAO BAN NGOI LECH VAI QUA MUC")


    #########Lưng##############
    if torso_state == "WARNING":
        warning_messages.append("BAN DANG GU LUNG")

    elif torso_state == "BAD":
        bad_messages.append("CANH BAO BAN NGOI GU LUNG QUA MUC")


    #########Cổ##############
    if neck_state == "WARNING":
        warning_messages.append("BAN DANG GU CO")

    elif neck_state == "BAD":
        bad_messages.append("CANH BAO BAN NGOI GU CO QUA MUC")
    


    #########Kết quả##############
    if shoulder_state is None or torso_state is None or neck_state is None:
        level = "NO_PERSON"
        messages = []
    elif bad_messages:
        level = "BAD_POSTURE"
        messages = bad_messages #Uu tien cac loi canh bao
    elif warning_messages   :
        level = "WARNING_POSTURE"
        messages = warning_messages
    else:
        level = "GOOD_POSTURE"
        messages = []

    ######### Join ##############
    if level == "GOOD_POSTURE":
        final_text = "TU THE DUNG"

    elif level == "NO_PERSON":
        final_text = "KHONG CO NGUOI"

    else:
        final_text = " + ".join(messages)
    

    # ==========================
    # Màu theo mức cảnh báo
    # ==========================
    if level == "GOOD_POSTURE":
        final_color = (0,255,0)

    elif level == "WARNING_POSTURE":
        final_color = (0,255,255)

    elif level == "BAD_POSTURE":
        final_color = (0,0,255)

    else:
        final_color =(123,234,111)

    cv2.putText(
            frame_front,
            f"{level}: {final_text}",
            (30,145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            final_color,
        2)
    
    if shouder_ang is not None:
        cv2.putText(
            frame_front,
            f"Shoulder: {shouder_ang:.1f}",
            (30,160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            final_color,
        2)

    

    if torso_ang is not None:
        cv2.putText(
            frame_side,
            f"torso: {torso_ang:.1f}",
            (30,180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            final_color,
        2)

    if neck_ang is not None:
        cv2.putText(
            frame_side,
            f"neck: {neck_ang:.1f}",
            (30,220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            final_color,
        2)

    current_status = (level, final_text)
    #############Xử lý gửi thông báo#############
    if  level == "GOOD_POSTURE":
            status_start_time = None  # Reset bộ đếm trì hoãn khi tư thế tốt
            
            if current_status != last_sent_status:
                send_snapshot(token, SESSION_ID, shouder_ang, torso_ang, neck_ang, level)
                send_message(token, level)
                last_sent_status = current_status
                
            # Reset trạng thái cảnh báo nguy hiểm liên tục 30s
            bad_start_time = None
            alert_sent = False


    else: # WARNING_POSTURE hoặc BAD_POSTURE
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



    if ret_front: cv2.imshow("Front Cam - Shoulder", frame_front)
    if ret_side: cv2.imshow("Side Cam - Torso & Neck", frame_side)



    if cv2.waitKey(1) == ord('q'):
        break


front_cam.release()
side_cam.release()
cv2.destroyAllWindows()