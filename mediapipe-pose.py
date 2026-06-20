import cv2
import mediapipe as mp
import math
import logging as logg
import time
import winsound
import requests
from collections import deque

front_cam = cv2.VideoCapture(0)
size_cam = cv2.VideoCapture("http://192.168.1.107:4747")
front_cam.set(3,1280)
front_cam.set(4,720)


#Initalize Mediapipe-pose
mpPose = mp.solutions.pose #lấy module Pose từ MediaPipe
pose = mpPose.Pose() #tạo object Pose để sử dụng
mpDraw = mp.solutions.drawing_utils #Khởi tạo lớp vẽ của Mediapipe

last_alert = 0
alert_sent = False


list = []
shouder_history = deque(maxlen = 15) #Luu toi da 15 phan tu
torso_history = deque(maxlen = 15)
neck_history = deque(maxlen = 15)


last_sent_status = None
status_start_time = None # Timer cho STATUS_DELAY (gửi snapshot)
bad_start_time = None # Timer riêng cho ALERT_DELAY (beep/alert)
STATUS_DELAY = 5
ALERT_DELAY = 30
token  = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0aGFuaG5ndXllbnNvbmpxa0BnbWFpbC5jb20iLCJyb2xlIjoiVVNFUiIsImlhdCI6MTc4MTkxOTg3NCwiZXhwIjoxNzgxOTIwNzc0fQ.yRAJsW-N96fsLZsuEdsFXvqD6_NBkW4haRN_zzDLUnE"
SESSION_ID = 4



def make_lm_timestaps(res): #Chứa toạ độ các điểm trên khung xương
    lm_list = []
    id = 0
    for lm in res.pose_landmarks.landmark:
        lm_list.append(f"idx:{id},x:{lm.x},y:{lm.y},z:{lm.z}")
        id+=1
    return lm_list 

def draw_lm_on_img(mpDraw,res,frame):
    mpDraw.draw_landmarks(frame,res.pose_landmarks,mpPose.POSE_CONNECTIONS) #Vẽ các đường nối
    id = 0
    #Vẽ các điểm nút
    for lm in res.pose_landmarks.landmark:
        h,w,c = frame.shape
        # print(id,lm)
        id+=1
        cx,cy = int(lm.x * w),int(lm.y * h)
        cv2.circle(frame,(cx,cy),5,(0,255,0),cv2.FILLED)
    return frame

def is_visibility(lm1,lm2):
    return lm1.visibility > 0.5 and lm2.visibility > 0.5


#Kiem tra ngồi lệch vai
def compare_Diff_Shoulder(res,frame):
    if not res.pose_landmarks:
        logg.error("Not found skeleton!")
        return None;
    lm = res.pose_landmarks.landmark
    h,w,c = frame.shape

    if not is_visibility(lm[11],lm[12]):
        return None

    left_shoulder_x,left_shoulder_y = int(lm[11].x * w),int(lm[11].y * h) #Toa do vai trai tren man anh
    right_shoulder_x,right_shoulder_y = int(lm[12].x * w),int(lm[12].y * h) #Toa do vai phai tren man anh
    diff_y = abs(left_shoulder_y- right_shoulder_y)
    diff_x = abs(left_shoulder_x - right_shoulder_x)
    if diff_x == 0: return None
    angle = math.degrees(math.atan2(abs(diff_y),abs(diff_x)))
    # ratio = diff_y / diff_x #Tinh ti le
    shouder_history.append(angle) #Save variable old
    smooth_ratio = sum(shouder_history) / len(shouder_history)
    return smooth_ratio

#Kiem tra ngoi gu lung
def torso_angle(res):
    if not res.pose_landmarks:
        logg.error("Not found skeleton!")
        return None
    lm = res.pose_landmarks.landmark
    shoulder_x,shoulder_y= (lm[11].x + lm[12].x) / 2,(lm[11].y + lm[12].y) / 2
    hip_x, hip_y= (lm[23].x + lm[24].x) / 2, (lm[23].y + lm[24].y) / 2
    delta_x, delta_y = shoulder_x - hip_x, shoulder_y - hip_y #Lấy hông làm gốc
    angle = math.degrees(math.atan2(abs(delta_x),abs(delta_y))) # Góc lệch so với phương thẳng đứng
    torso_history.append(angle)
    smooth_angle = sum(torso_history) / len(torso_history)
    return smooth_angle

#Kiem tra ngoi cúi đầu
def neck_angle(res):
    if not res.pose_landmarks:
        logg.error("Not found skeleton!")
        return None
    lm = res.pose_landmarks.landmark

    nose_x,nose_y = lm[0].x,lm[0].y

    shoulder_x = (lm[11].x + lm[12].x) / 2
    shoulder_y = (lm[11].y + lm[12].y) / 2
    
    hip_x = (lm[23].x + lm[24].x) / 2
    hip_y = (lm[23].y + lm[24].y) / 2
    
    v1_x,v1_y = nose_x - shoulder_x, nose_y - shoulder_y #vector shoulder,nose
    v2_x,v2_y = hip_x - shoulder_x,hip_y - shoulder_y #vector shoulder,hip

    scalar_product = v1_x * v2_x + v1_y * v2_y #a*b = x1x2 + y1y2
    len_v1,len_v2 = math.sqrt(v1_x**2 + v1_y**2),math.sqrt(v2_x**2 + v2_y**2) #Tinh do dai vector

    if len_v1 == 0 or len_v2 == 0:
        return None

    angle_cos = max(-1,min(1,scalar_product / (len_v1 * len_v2) ))


    angle_result = math.degrees(math.acos(angle_cos)) #Tinh goc 2 vector

    # Chuyển về độ lệch cổ
    neck_devitaion = abs(180 - angle_result)
    neck_history.append(neck_devitaion)
    smooth_neck = sum(neck_history) / len(neck_history)
    return smooth_neck

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
    ret_front,frame_front = front_cam.read()
    ret_side, frame_side = size_cam.read()

    if ret:
        frame = cv2.flip(frame,1)
        frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)    
        pose_reslut = pose.process(frameRGB) #Đưa ảnh vào cho AI phân tích
        if pose_reslut.pose_landmarks:

            list = make_lm_timestaps(pose_reslut)
            frame = draw_lm_on_img(mpDraw,pose_reslut,frame)

            warning_count = 0 #Đếm cảnh báo
            bad_count = 0 #Đếm cảnh báo tệ


            #Kiểm tra ngồi lệch vai
            shouder_ang = compare_Diff_Shoulder(pose_reslut,frame)
            if shouder_ang is not None:
                if shouder_ang < 5:
                    text = "GOOD"
                    color = (0,255,0)
                elif shouder_ang < 10:
                    text = "WARNING"
                    color = (0, 255, 255)
                    warning_count += 1
                else:
                    text = "BAD"
                    color = (0, 0, 255)
                    bad_count += 1
                cv2.putText(frame,f"Rad: {shouder_ang:.1f}, Shoulder: {text}",(50,100),cv2.FONT_HERSHEY_COMPLEX,1,color,2) 


            #Kiểm tra gù lưng
            torso_ang = torso_angle(pose_reslut)
            if torso_ang  is not None:
                if torso_ang  < 8:
                    text = "GOOD"
                    color = (0, 255, 0)
                elif torso_ang  < 15:
                    text = "WARNING"
                    color = (0, 255, 255)
                    warning_count += 1
                else:
                    text = "BAD"
                    color = (0, 0, 255)
                    bad_count += 1
                cv2.putText(frame,f"Rad: {torso_ang:.1f}, Torso: {text}",(50,160),cv2.FONT_HERSHEY_COMPLEX,1,color,2)


            #Kiểm tra gù cổ
            neck_ang  = neck_angle(pose_reslut)
            if neck_ang  is not None:
                if neck_ang  < 15:
                    text = "GOOD"
                    color = (0,255,0)
                elif neck_ang  < 25:
                    text = "WARNING"
                    color = (0,255,255)
                    warning_count += 1
                else:
                    text = "BAD"
                    color = (0,0,255)
                    bad_count += 1
                cv2.putText(frame,f"Rad: {neck_ang:.1f}, Neck: {text}",(50,220),cv2.FONT_HERSHEY_COMPLEX,1,color,2)
         
         
           # ======================== Kết quả tổng hợp ========================

            if bad_count > 0:
                final_text = "BAD_POSTURE"
                color = (0,0,255)
            
            elif warning_count > 0:
                final_text = "WARNING_POSTURE"
                color = (0,255,255)

            else:
                final_text = "GOOD_POSTURE"
                color = (0,255,0)


             # Hiển thị status
            cv2.putText(frame,final_text,(50,280),
                        cv2.FONT_HERSHEY_COMPLEX,1.2,color,3)

            # ======================== Reset khi tư thế tốt lại ====================
            if(final_text != "BAD_POSTURE"):
                alert_sent = False
                bad_start_time = None

            if final_text == "GOOD_POSTURE":
                status_start_time = None # Không cần đếm delay khi đã tốt

            # ======================== Gửi snapshot ========================
            if final_text == "GOOD_POSTURE":
                if(final_text != last_sent_status):
                    send_snapshot(token,SESSION_ID,shouder_ang,torso_ang,neck_ang,final_text)
                    send_message(token,final_text)
                last_sent_status = final_text
                status_start_time = None

            else:
                # WARNING hoặc BAD: đợi STATUS_DELAY giây rồi mới gửi
                if final_text != last_sent_status:
                    if status_start_time is None:
                        status_start_time = time.time() #Bat dau dem
                    posture_duration = time.time() - status_start_time
                    cv2.putText(frame, f"Delay: {int(posture_duration)}s", (50, 340),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    
                    if posture_duration >= STATUS_DELAY:
                        send_snapshot(token,SESSION_ID,shouder_ang,torso_ang,neck_ang,final_text)
                        send_message(token,final_text)
                        last_sent_status = final_text
                        status_start_time = None # Reset sau khi gửi
                else:
                    status_start_time = None



            # ============== BAD ALERT (Beep sau 30s) =====================
            if final_text == "BAD_POSTURE":
                if bad_start_time is None:
                    bad_start_time = time.time()
                bad_duration = time.time() - bad_start_time
                cv2.putText(frame, f"Bad Duration: {int(bad_duration)}s", (50, 380),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                if bad_duration >= ALERT_DELAY and not alert_sent:
                    print("Vui lòng ngồi đúng tư thế!")
                    send_alert(token,SESSION_ID,final_text)
                    alert_sent = True

            # ======================== Debug info ========================
            cv2.putText(frame, f"Last sent: {last_sent_status}", (50, 450),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (93, 32, 255), 2)
            cv2.putText(frame, f"Final Text: {final_text}", (50, 490),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (99, 38, 255), 2)
        cv2.imshow("Posture Monitor",frame)
        if cv2.waitKey(1) == ord('q'): 
            break
        
cap.release()
cv2.destroyAllWindows()