import cv2
import mediapipe as mp
import time
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
class Process:
    def __init__(self,img):
        self.hands=hands
        self.mpDraw=mpDraw
        self.img = img
    def process_frame(self):
        start_time = time.time()

        # 获取图像宽高
        h, w = self.img.shape[0], self.img.shape[1]

        # 水平镜像翻转图像，使图中左右手与真实左右手对应
        # 参数 1：水平翻转，0：竖直翻转，-1：水平和竖直都翻转
        img = cv2.flip(self.img, 1)
        # BGR转RGB
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 将RGB图像输入模型，获取预测结果
        results = self.hands.process(img_RGB)

        if results.multi_hand_landmarks:  # 如果有检测到手

            handness_str = ''
            index_finger_tip_str = ''
            for hand_idx in range(len(results.multi_hand_landmarks)):

                # 获取并输出该手的21个关键点坐标
                hand_21 = results.multi_hand_landmarks[hand_idx]
                for index, landmarks in enumerate(hand_21.landmark):
                    print(index, landmarks)
                # 可视化关键点及骨架连线
                mpDraw.draw_landmarks(img, hand_21, mp_hands.HAND_CONNECTIONS)

                # 记录左右手信息
                temp_handness = results.multi_handedness[hand_idx].classification[0].label
                handness_str += '{}:{} '.format(hand_idx, temp_handness)

                # 获取手腕根部深度坐标
                cz0 = hand_21.landmark[0].z

                for i in range(21):  # 遍历该手的21个关键点

                    # 获取3D坐标
                    cx = int(hand_21.landmark[i].x * w)
                    cy = int(hand_21.landmark[i].y * h)
                    cz = hand_21.landmark[i].z
                    depth_z = cz0 - cz

                    # 用圆的半径反映深度大小
                    radius = max(int(3.5 * (1 + depth_z * 5)), 0)

                    if i == 0:  # 手腕
                        img = cv2.circle(img, (cx, cy), radius, (0, 215, 255), -1)
                    if i in [1, 5, 9, 13, 17]:  # 指根
                        img = cv2.circle(img, (cx, cy), radius, (87, 201, 0), -1)
                    if i in [2, 6, 10, 14, 18]:  # 第一指节
                        img = cv2.circle(img, (cx, cy), radius, (208, 224, 64), -1)
                    if i in [3, 7, 11, 15, 19]:  # 第二指节
                        img = cv2.circle(img, (cx, cy), radius, (225, 105, 65), -1)
                    if i in [4, 8, 12, 16, 20]:  # 指尖
                        img = cv2.circle(img, (cx, cy), radius, (226, 43, 138), -1)

            scaler = 1
            img = cv2.putText(img, handness_str, (25 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1 * scaler,
                              (255, 255, 255), 2 * scaler)
            img = cv2.putText(img, index_finger_tip_str, (25 * scaler, 150 * scaler), cv2.FONT_HERSHEY_SIMPLEX,
                              1 * scaler, (255, 255, 255), 2 * scaler)

            # 记录该帧处理完毕的时间
            end_time = time.time()
            # 计算每秒处理图像帧数FPS
            FPS = 1 / (end_time - start_time)

            # 在图像上写FPS数值，参数依次为：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
            img = cv2.putText(img, 'FPS: ' + str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX,
                              1 * scaler, (255, 255, 255), 2 * scaler)
        return img
