import RPi.GPIO as GPIO
import time
import cv2
import numpy as np
import copy
from pynput import keyboard


class RelayController:  # 电磁铁控制
    def __init__(self, relay_pin, debounce_time=0.5):
        """
        初始化继电器控制类
        :param relay_pin: 继电器连接的 GPIO 引脚
        :param debounce_time: 控制继电器的状态保持时间（秒）
        """
        self.relay_pin = relay_pin
        self.debounce_time = debounce_time

        # 设置 GPIO 模式为 BCM
        GPIO.setmode(GPIO.BCM)

        # 设置引脚为输出模式
        GPIO.setup(self.relay_pin, GPIO.OUT)

    def turn_on(self):
        """开启继电器"""
        print("继电器开启")
        GPIO.output(self.relay_pin, GPIO.HIGH)  # 继电器开启

    def turn_off(self):
        """关闭继电器"""
        print("继电器关闭")
        GPIO.output(self.relay_pin, GPIO.LOW)  # 继电器关闭

    def run(self):
        """控制继电器的状态"""
        try:
            while True:
                self.turn_on()
                time.sleep(self.debounce_time)  # 保持状态

                self.turn_off()
                time.sleep(self.debounce_time)  # 保持状态

        except KeyboardInterrupt:
            print("程序终止，清理 GPIO 设置")

        finally:
            self.cleanup()

    def cleanup(self):
        """清理 GPIO 设置"""
        GPIO.cleanup()


class BHSBFL:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def find_grid_coordinates(self):
        edges = cv2.Canny(self.gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        contours_info = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

        if contours:
            contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            width, height = rect[1]
            cell_width = width / 3
            cell_height = height / 3
            box = sorted(box, key=lambda x: (x[1], x[0]))
            coordinates = [(i * 3 + j + 1, (int(box[0][0] + j * cell_width + cell_width / 2),
                                            int(box[0][1] + i * cell_height + cell_height / 2)))
                           for i in range(3) for j in range(3)]
            return coordinates
        return []

    def find_corners(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 30, 150)
        contours_info = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            return box.tolist()
        return []


class ChessPieceDetector(BHSBFL):
    def __init__(self, image_path):
        super().__init__(image_path)
        self.coordinates = self.find_grid_coordinates()
        self.corner_coordinates = self.find_corners()

    def detect_pieces(self):
        if not self.corner_coordinates:
            print("未找到九宫格角点坐标信息。")
            return []

        mask = np.zeros(self.gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(self.corner_coordinates)], 255)

        masked_gray = cv2.bitwise_and(self.gray, self.gray, mask=mask)

        blurred = cv2.GaussianBlur(masked_gray, (9, 9), 2)

        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10,
                                   maxRadius=45)

        piece_positions = []

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            # 为每个检测到的圆准备盒子和分数
            boxes = np.array([[x - r, y - r, x + r, y + r] for (x, y, r) in circles])
            scores = np.array([1] * len(circles))  # 这里简单设定每个圆的得分为1

            # 执行非极大抑制
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.5, nms_threshold=0.5)

            # 解析经过非极大抑制后的圆
            if len(indices) > 0:
                for i in indices:
                    i = i[0]  # 提取索引
                    x, y, r = circles[i]

                    # 获取颜色
                    color = masked_gray[y, x]
                    piece_color = "黑色" if color < 128 else "白色"

                    closest_distance = float('inf')
                    closest_number = None
                    for number, (cx, cy) in self.coordinates:
                        distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_number = number

                    if closest_number is not None:
                        piece_positions.append((closest_number, piece_color))
                        # 标记棋子的位置
                        point_color = (0, 0, 255) if piece_color == "黑色" else (0, 255, 0)  # 红色或绿色圆点
                        cv2.circle(self.image, (x, y), 10, point_color, -1)

        piece_positions.sort(key=lambda x: x[0])

        return piece_positions


def main9(image_path):
    class BHSBFL:
        def __init__(self, image_path):
            self.image = cv2.imread(image_path)
            self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        def find_grid_coordinates(self):
            edges = cv2.Canny(self.gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
            contours_info = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

            if contours:
                contour = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                width, height = rect[1]
                cell_width = width / 3
                cell_height = height / 3
                box = sorted(box, key=lambda x: (x[1], x[0]))
                coordinates = [(i * 3 + j + 1, (int(box[0][0] + j * cell_width + cell_width / 2),
                                                int(box[0][1] + i * cell_height + cell_height / 2)))
                               for i in range(3) for j in range(3)]
                return coordinates
            return []

        def find_corners(self):
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            edged = cv2.Canny(gray, 30, 150)
            contours_info = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(largest_contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                return box.tolist()
            return []

    class ChessPieceDetector(BHSBFL):
        def __init__(self, image_path):
            super().__init__(image_path)
            self.coordinates = self.find_grid_coordinates()
            self.corner_coordinates = self.find_corners()

        def detect_pieces(self):
            if not self.corner_coordinates:
                print("未找到九宫格角点坐标信息。")
                return []

            mask = np.zeros(self.gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(self.corner_coordinates)], 255)

            masked_gray = cv2.bitwise_and(self.gray, self.gray, mask=mask)

            blurred = cv2.GaussianBlur(masked_gray, (9, 9), 2)

            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30,
                                       minRadius=10, maxRadius=45)

            piece_positions = []

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")

                # 为每个检测到的圆准备盒子和分数
                boxes = np.array([[x - r, y - r, x + r, y + r] for (x, y, r) in circles])
                scores = np.array([1] * len(circles))  # 这里简单设定每个圆的得分为1

                # 执行非极大抑制
                indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.5, nms_threshold=0.5)

                # 解析经过非极大抑制后的圆
                if len(indices) > 0:
                    for i in indices:
                        i = i[0]  # 提取索引
                        x, y, r = circles[i]

                        # 获取颜色
                        color = masked_gray[y, x]
                        piece_color = "黑色" if color < 128 else "白色"

                        closest_distance = float('inf')
                        closest_number = None
                        for number, (cx, cy) in self.coordinates:
                            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                            if distance < closest_distance:
                                closest_distance = distance
                                closest_number = number

                        if closest_number is not None:
                            piece_positions.append((closest_number, piece_color))
                            # 标记棋子的位置
                            point_color = (0, 0, 255) if piece_color == "黑色" else (0, 255, 0)  # 红色或绿色圆点
                            cv2.circle(self.image, (x, y), 10, point_color, -1)

            piece_positions.sort(key=lambda x: x[0])

            return piece_positions

        def print_corners(self):
            if self.corner_coordinates:
                print("九宫格四个角的坐标信息:")
                for i, (x, y) in enumerate(self.corner_coordinates):
                    print(f"角点 {i + 1}: (x: {x}, y: {y})")
            else:
                print("未能找到九宫格角点坐标信息。")

    def game_over_condition(mark):
        sum_row = mark.sum(axis=1)
        sum_column = mark.sum(axis=0)
        main_diagonal = mark[0, 0] + mark[1, 1] + mark[2, 2]
        back_diagonal = mark[0, 2] + mark[1, 1] + mark[2, 0]

        if any(sum_row == -3) or any(sum_column == -3) or main_diagonal == -3 or back_diagonal == -3:
            return '玩家胜'
        if any(sum_row == 3) or any(sum_column == 3) or main_diagonal == 3 or back_diagonal == 3:
            return '电脑胜'
        if not np.any(mark == 0):
            return '和棋'
        return None

    def Key_points(M, t):
        x = []
        s_r = M.sum(axis=1)
        s_c = M.sum(axis=0)
        m_d = M[0, 0] + M[1, 1] + M[2, 2]
        b_d = M[0, 2] + M[1, 1] + M[2, 0]
        for i in range(3):
            for j in range(3):
                if M[i][j] == 0:
                    c = 0
                    if i == j and m_d == t: c += 1
                    if i + j == 2 and b_d == t: c += 1
                    if s_r[i] == t: c += 1
                    if s_c[j] == t: c += 1
                    if c >= 2:
                        x.append([i, j])
        return x

    def next_move(board):
        mark = np.array(board)
        sum_row = mark.sum(axis=1)
        sum_column = mark.sum(axis=0)
        main_diagonal = mark[0, 0] + mark[1, 1] + mark[2, 2]
        back_diagonal = mark[0, 2] + mark[1, 1] + mark[2, 0]

        for i in range(3):
            if sum_row[i] == -3 or sum_column[i] == -3:
                return '玩家胜'
        if main_diagonal == -3 or back_diagonal == -3:
            return '玩家胜'

        for i in range(3):
            if sum_row[i] == 2:
                return [i, np.where(mark[i] == 0)[0][0]]
            if sum_column[i] == 2:
                return [np.where(mark[:, i] == 0)[0][0], i]
        if main_diagonal == 2:
            for i in range(3):
                if mark[i, i] == 0:
                    return [i, i]
        if back_diagonal == 2:
            for i in range(3):
                if mark[i, 2 - i] == 0:
                    return [i, 2 - i]

        for i in range(3):
            if sum_row[i] == -2:
                return [i, np.where(mark[i] == 0)[0][0]]
            if sum_column[i] == -2:
                return [np.where(mark[:, i] == 0)[0][0], i]
        if main_diagonal == -2:
            for i in range(3):
                if mark[i, i] == 0:
                    return [i, i]
        if back_diagonal == -2:
            for i in range(3):
                if mark[i, 2 - i] == 0:
                    return [i, 2 - i]

        K = Key_points(mark, 1)
        if K:
            return K[0]

        K = Key_points(mark, -1)
        if K:
            return K[0]

        for i in range(3):
            for j in range(3):
                if mark[i][j] == 0:
                    mark_new = copy.deepcopy(mark)
                    mark_new[i][j] = 1
                    K = Key_points(mark_new, 1)
                    if len(K) >= 2:
                        return [i, j]

        for i in range(3):
            for j in range(3):
                if mark[i][j] == 0:
                    mark_new = copy.deepcopy(mark)
                    mark_new[i][j] = -1
                    K = Key_points(mark_new, -1)
                    if len(K) >= 2:
                        return [i, j]

        for i in range(3):
            for j in range(3):
                if mark[i][j] == 0:
                    return [i, j]

        return '和棋'

    detector = ChessPieceDetector(image_path)
    piece_positions = detector.detect_pieces()
    board = [[0 for _ in range(3)] for _ in range(3)]

    for number, color in piece_positions:
        row = (number - 1) // 3
        col = (number - 1) % 3
        if color == "黑色":
            board[row][col] = 1
        elif color == "白色":
            board[row][col] = -1
        print(f"棋子位置编号: {number}, 颜色: {color}")

    move = next_move(board)
    if isinstance(move, list):
        grid_number = move[0] * 3 + move[1] + 1
        print(f"电脑下一步应该走的位置: {grid_number}")
    else:
        print(f"游戏结果: {move}")

    return move


class ChessBoardDetector:
    def __init__(self, image_path):
        # 读取图像并转换为灰度图
        self.image = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.coordinates = self.find_grid_coordinates()
        self.corner_coordinates = self.find_corners()
        self.masked_image = self.image.copy()

    def find_grid_coordinates(self):
        # 应用边缘检测
        edges = cv2.Canny(self.gray, 50, 150, apertureSize=3)

        # 使用霍夫变换检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        # 寻找轮廓
        _, contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 选择最大的轮廓，这通常是九宫格的外框
            contour = max(contours, key=cv2.contourArea)

            # 获取最小的外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # 计算每个格子的中心并编号
            width, height = rect[1]
            cell_width = width / 3
            cell_height = height / 3

            # 确保顶点顺序为左上、右上、右下、左下
            box = sorted(box, key=lambda x: (x[1], x[0]))

            coordinates = []
            for i in range(3):
                for j in range(3):
                    # 计算格子中心
                    center_x = int(box[0][0] + j * cell_width + cell_width / 2)
                    center_y = int(box[0][1] + i * cell_height + cell_height / 2)

                    # 返回编号和中心坐标
                    coordinates.append((i * 3 + j + 1, (center_x, center_y)))

            return coordinates
        else:
            print("未找到轮廓")
            return []

    def find_corners(self):
        edged = cv2.Canny(self.gray, 30, 150)
        _, contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            return box.tolist()
        return []

    def mask_rectangle(self):
        if self.corner_coordinates:
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            points = np.array(self.corner_coordinates, dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)

            mask_inv = cv2.bitwise_not(mask)
            self.masked_image = cv2.bitwise_and(self.image, self.image, mask=mask_inv)
        else:
            print("未能找到九宫格角点坐标信息。")

    def detect_pieces(self):
        blurred = cv2.GaussianBlur(cv2.cvtColor(self.masked_image, cv2.COLOR_BGR2GRAY), (9, 9), 2)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2,
                                   minDist=30, param1=50, param2=30,
                                   minRadius=10, maxRadius=45)

        piece_positions = []

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                color = cv2.cvtColor(self.masked_image, cv2.COLOR_BGR2GRAY)[y, x]
                piece_positions.append(((x, y), color))

        piece_positions.sort(key=lambda pos: (pos[0][1], pos[0][0]))
        return piece_positions

    def save_pieces_info(self, index):
        piece_positions = self.detect_pieces()
        pieces_info = {}
        for counter, ((x, y), color) in enumerate(piece_positions, start=1):
            pieces_info[counter] = {'coordinates': (x, y), 'color': color}

        # 生成文件名
        filename = f'pieces_info_{index}.txt'

        # 将信息保存到文件中，使用 UTF-8 编码
        with open(filename, 'w', encoding='utf-8') as file:
            for piece_id, info in pieces_info.items():
                file.write(f"{info['coordinates']}\n")

        return filename


class ArmControl:
    def __init__(self):
        GPIO.setwarnings(False)
        # 初始化舵机引脚
        self.servos = {
            'base': 16,
            'shoulder': 20,
            'elbow': 21
        }

        # 设定每个九宫格位置对应的舵机角度
        self.position_angles = {
            0: (55,100, 80),
            1: (91, 106, 35),  # 位置1: 基座97°，肩部14°，肘部0° x
            2: (86,106, 38),  # 位置2: 基座90°，肩部14°，肘部0° x
            3: (81, 106, 36),  # 位置3: 基座82°，肩部14°，肘部0° x
            4: (91,102, 68),   # 位置4: 基座97°，肩部5°，肘部3° x
            5: (86, 102, 68),   # 位置5: 基座90°，肩部0°，肘部78° x
            6: (80, 102, 68),   # 位置6: 基座82°，肩部5°，肘部3° x
            7: (91, 101, 88),  # 位置7: 基座99°，肩部0°，肘部45° x
            8: (85, 102, 88),  # 位置8: 基座90°，肩部0°，肘部45°  x
            9: (78, 101, 88) ,  # 位置9: 基座80°，肩部0°，肘部46° x
            'A1':(98, 106,46),  # 位置1: 基座97°，肩部14°，肘部0° x
            'B1':(100,104,70),  # 位置1: 基座97°，肩部14°，肘部0° x
            'C1':(101,103,88),  # 位置1: 基座97°，肩部14°，肘部0° x
            'D1':(103, 103, 100),  # 位置1: 基座97°，肩部14°，肘部0°x
            'A2':(73, 106,47),  # 位置1: 基座97°，肩部14°，肘部0° x
            'B2':(72, 104, 73),  # 位置1: 基座97°，肩部14°，肘部0°x
            'C2':(69, 104, 90),  # 位置1: 基座97°，肩部14°，肘部0°x
            'D2':(66, 103, 104),  # 位置1: 基座97°，肩部14°，肘部0°x
        }

        # 设置GPIO模式
        GPIO.setmode(GPIO.BCM)

        # 设置舵机引脚
        self.pwm = {}
        self.last_time = 0  # 上一次设置时间
        self.debounce_time = 0.03  # 防抖时间（秒）

        for servo in self.servos.values():
            GPIO.setup(servo, GPIO.OUT)
            self.pwm[servo] = GPIO.PWM(servo, 50)  # 50Hz
            self.pwm[servo].start(0)

    def set_servo_angle(self, pwm, target_angle, speed):
        """将舵机平滑旋转到指定角度，并实现防抖逻辑"""
        current_time = time.time()  # 获取当前时间

        # 防抖逻辑
        if current_time - self.last_time < self.debounce_time:
            return  # 如果距离上次设置时间小于防抖时间，则不执行

        # 计算占空比
        duty = self.angle_to_duty_cycle(target_angle)

        # 确保占空比在合理范围内
        if duty < 0:
            duty = 0
        elif duty > 12:  # 假设最大是180度
            duty = 12

        pwm.ChangeDutyCycle(duty)
        time.sleep(speed)  # 按速度控制时间延迟
        pwm.ChangeDutyCycle(0)  # 停止PWM信号
        self.last_time = current_time  # 更新最后设置时间

    def move_servo(self, servo_name, angle):
        """指定舵机移动到指定角度"""
        self.set_servo_angle(self.pwm[self.servos[servo_name]], angle, 0.5)  # 设置速度为0.5秒

    def move_to_position(self, position):
        """移动机械臂到指定棋盘位置（1-9）"""
        if position in self.position_angles:
            angles = self.position_angles[position]  # 获取角度元组
            self.move_servo('base', angles[0])  # 移动基座
            self.move_servo('shoulder', angles[1])  # 移动肩部
            self.move_servo('elbow', angles[2])  # 移动肘部
        else:
            print("无效的位置编号！")

    def remove_to_position(self, position):
        """移动机械臂到指定棋盘位置（1-9）"""
        if position in self.position_angles:
            angles = self.position_angles[position]  # 获取角度元组
            self.move_servo('shoulder', angles[1])  # 移动基座
            time.sleep(0.5)
            self.move_servo('elbow', angles[2])
            time.sleep(1)
            self.move_servo('base', angles[0])
        else:
            print("无效的位置编号！")

    def angle_to_duty_cycle(self, angle):
        """将角度转换为PWM占空比（0到100之间）"""
        return (angle / 18) + 2  # 示例转换，具体根据舵机类型调整

    def cleanup(self):
        """清理GPIO设置"""
        for pwm in self.pwm.values():
            pwm.stop()
        GPIO.cleanup()


class ArmControlA(ArmControl):
    def __init__(self):
        super().__init__()
        self.position_angles.update({
            11: (101, 125, 55),
            12: (78, 125, 83),
            13: (101,103,88),
            14: (103, 103, 100)
        })
        self.sequence_positions = [11, 12, 13, 14]
        self.current_sequence_index = 0

    def move_to_next_sequence_position(self):
        if self.current_sequence_index < len(self.sequence_positions):
            position = self.sequence_positions[self.current_sequence_index]
            self.move_to_position(position)
            self.current_sequence_index += 1
        else:
            print("序列完成。重置到起始位置。")
            self.current_sequence_index = 0
            self.move_to_next_sequence_position()


class ArmControlB(ArmControl):  # 第s问 白色棋子抓取序列 待调整
    def __init__(self):
        super().__init__()
        self.position_angles.update({
            21: (98, 106,46),
            22: (100,104,70),
            23: (101,103,88),
            24: (103, 103, 100)
        })
        self.sequence_positions = [21, 22, 23, 24]
        self.current_sequence_index = 0

    def move_to_next_sequence_position(self):
        if self.current_sequence_index < len(self.sequence_positions):
            position = self.sequence_positions[self.current_sequence_index]
            self.move_to_position(position)
            self.current_sequence_index += 1
        else:
            print("序列完成。重置到起始位置。")
            self.current_sequence_index = 0
            self.move_to_next_sequence_position()


class ArmControlC(ArmControl):  # 第五问 白色棋子抓取序列 待调整
    def __init__(self):
        super().__init__()
        self.position_angles.update({
            31: (101, 125, 55),  # 位置2: 基座101°，肩部125°，肘部55°
            32: (102, 125, 70),  # 位置3: 基座102°，肩部125°，肘部70°
            33: (104, 125, 83),  # 位置4: 基座104°，肩部125°，肘部83°
            34: (106, 125, 100)  # 位置5: 基座106°，肩部125°，肘部100°
        })
        self.sequence_positions = [31, 32, 33, 34]
        self.current_sequence_index = 0

    def move_to_next_sequence_position(self):
        if self.current_sequence_index < len(self.sequence_positions):
            position = self.sequence_positions[self.current_sequence_index]
            self.move_to_position(position)
            self.current_sequence_index += 1
        else:
            print("序列完成。重置到起始位置。")
            self.current_sequence_index = 0
            self.move_to_next_sequence_position()


def take_photo(filename='1.jpg'):
    """调用摄像头拍照并保存为指定文件名"""
    cap = cv2.VideoCapture(0)  # 使用第一个摄像头
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    ret, frame = cap.read()
    if ret:
        cv2.imwrite(filename, frame)
        print(f"图像已保存为 {filename}")
    else:
        print("无法拍摄图像")

    cap.release()


def take_photo(filename='1.jpg'):
    """调用摄像头拍照并保存为指定文件名"""
    cap = cv2.VideoCapture(0)  # 使用第一个摄像头
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    ret, frame = cap.read()
    if ret:
        cv2.imwrite(filename, frame)
        print(f"图像已保存为 {filename}")
    else:
        print("无法拍摄图像")

    cap.release()
def on_press(key):
    global capture_next_image
    try:
        if key.char == 'v':
            capture_next_image = True
    except AttributeError:
        pass





def on_release(key):
    if key == keyboard.Key.esc:
        # 退出程序的条件
        return False  # 停止监听


import cv2
import numpy as np

def main7(image_path):
    class BHSBFL:
        def __init__(self, image_path):
            # 读取图像并转换为灰度图
            self.image = cv2.imread(image_path)
            self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        def find_grid_coordinates(self):
            # 应用边缘检测
            edges = cv2.Canny(self.gray, 50, 150, apertureSize=3)

            # 使用霍夫变换检测直线
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

            # 寻找轮廓
            contours_info = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

            if contours:
                # 选择最大的轮廓，这通常是九宫格的外框
                contour = max(contours, key=cv2.contourArea)

                # 获取最小的外接矩形
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                # 计算每个格子的中心并编号
                width, height = rect[1]
                cell_width = width / 3
                cell_height = height / 3

                # 确保顶点顺序为左上、右上、右下、左下
                box = sorted(box, key=lambda x: (x[1], x[0]))

                coordinates = []
                for i in range(3):
                    for j in range(3):
                        # 计算格子中心
                        center_x = int(box[0][0] + j * cell_width + cell_width / 2)
                        center_y = int(box[0][1] + i * cell_height + cell_height / 2)

                        # 返回编号和中心坐标
                        coordinates.append((i * 3 + j + 1, (center_x, center_y)))

                return coordinates
            else:
                print("未找到轮廓")
                return []

    class ChessPieceDetector(BHSBFL):
        def __init__(self, image_path):
            super().__init__(image_path)
            self.image_path = image_path  # 存储图像路径
            self.coordinates = self.find_grid_coordinates()
            self.corner_coordinates = self.find_corners()
            self.image = cv2.imread(image_path)  # 读取图像

        def find_corners(self):
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            edged = cv2.Canny(gray, 30, 150)

            contours_info = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(largest_contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                return box.tolist()
            return []

        def detect_pieces(self):
            if not self.corner_coordinates:
                print("未找到九宫格角点坐标信息。")
                return []

            mask = np.zeros(self.gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(self.corner_coordinates)], 255)

            masked_gray = cv2.bitwise_and(self.gray, self.gray, mask=mask)

            blurred = cv2.GaussianBlur(masked_gray, (9, 9), 2)

            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30,
                                       minRadius=10, maxRadius=45)

            piece_positions = []

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")

                # 为每个检测到的圆准备盒子和分数
                boxes = np.array([[x - r, y - r, x + r, y + r] for (x, y, r) in circles])
                scores = np.array([1] * len(circles))  # 这里简单设定每个圆的得分为1

                # 执行非极大抑制
                indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.5, nms_threshold=0.5)

                # 解析经过非极大抑制后的圆
                if len(indices) > 0:
                    for i in indices:
                        i = i[0]  # 提取索引
                        x, y, r = circles[i]

                        # 获取颜色
                        color = masked_gray[y, x]
                        piece_color = "黑色" if color < 128 else "白色"

                        closest_distance = float('inf')
                        closest_number = None
                        for number, (cx, cy) in self.coordinates:
                            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                            if distance < closest_distance:
                                closest_distance = distance
                                closest_number = number

                        if closest_number is not None:
                            piece_positions.append((closest_number, piece_color))

            piece_positions.sort(key=lambda x: x[0])

            return piece_positions

        def print_corners(self):
            if self.corner_coordinates:
                print("九宫格四个角的坐标信息:")
                for i, (x, y) in enumerate(self.corner_coordinates):
                    print(f"角点 {i + 1}: (x: {x}, y: {y})")
            else:
                print("未能找到九宫格角点坐标信息。")

    detector = ChessPieceDetector(image_path)
    piece_positions = detector.detect_pieces()

    board = [[0 for _ in range(3)] for _ in range(3)]

    for number, color in piece_positions:
        row = (number - 1) // 3
        col = (number - 1) % 3
        if color == "黑色":
            board[row][col] = 1
        elif color == "白色":
            board[row][col] = -1
        print(f"棋子位置编号: {number}, 颜色: {color}")

    detector.print_corners()

    return board


def convert_to_number(position):
    row, col = position
    return row * 3 + col + 1

def main8():
    global i  # 让 i 在函数中可用
    i = 0

    # 创建 ArmControlB 实例
    arm = ArmControlB()
    # 初始化电磁继电器
    relay_controller = RelayController(relay_pin=17)

    # 调用 main7 函数进行棋盘初始化
    board = main7('/home/pi/AI-chess/04/1.png')

    # 打印棋盘的初始化结果
    print("棋盘初始化结果:")
    for row in board:
        print(row)

    # 进行棋盘状态的初步判断
    if all(cell == 0 for row in board for cell in row):
        try:
            arm.move_to_position(21)
            time.sleep(5)

            print("打开电磁继电器")
            relay_controller.turn_on()
            time.sleep(1)

            print("移动到位置 5")
            arm.move_to_position(5)
            time.sleep(2)

            print("关闭电磁继电器")
            relay_controller.turn_off()
            time.sleep(2)

            print("返回初始位置")
            arm.move_to_position(0)
            time.sleep(2)

        except Exception as e:
            print(f"发生错误: {e}")

    def on_press(key):
        try:
            if key.char == 'v':
                # 处理按下 'v' 键的逻辑
                handle_v_key()
        except AttributeError:
            pass

    def handle_v_key():
        global i  # 使得可以访问外部变量 i

        try:
            # 调取摄像头并拍摄图片
            camera = cv2.VideoCapture(0)
            ret, frame = camera.read()
            if ret:
                image_path = f'/home/pi/AI-chess/04/{i + 2}.png'
                cv2.imwrite(image_path, frame)

                # 释放摄像头资源
                camera.release()
                cv2.destroyAllWindows()

                # 调用 main9 函数获取黑棋下一步应该走的位置
                move = main9(image_path)
                co_move = convert_to_number(move)
                print(f"黑棋下一步应该走的位置: {move}")

                # 更新棋盘上的棋子位置
                piece_positions = ChessPieceDetector(image_path).detect_pieces()
                board = [[0 for _ in range(3)] for _ in range(3)]
                for number, color in piece_positions:
                    row = (number - 1) // 3
                    col = (number - 1) % 3
                    if color == "黑色":
                        board[row][col] = 1
                    elif color == "白色":
                        board[row][col] = -1
                    print(f"棋子位置编号: {number}, 颜色: {color}")

                # 计算目标位置
                target_position = 22 + i  # 22, 23, 24, 25

                # 执行机械臂运动
                print(f"移动到目标位置 {target_position}")
                arm.move_to_position(target_position)
                time.sleep(2)

                # 打开继电器
                print("打开电磁继电器")
                relay_controller.turn_on()
                time.sleep(2)

                # 移动机械臂到 move 位置
                if isinstance(move, int) and 1 <= move <= 9:
                    print(f"移动到位置 {move}")
                arm.move_to_position(co_move)
                time.sleep(2)

                # 关闭继电器
                print("关闭电磁继电器")
                relay_controller.turn_off()

                # 延时后返回到0位置
                time.sleep(2)
                print("返回初始位置")
                arm.move_to_position(0)
            else:
                print("无法获取摄像头图像。")
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            if 'camera' in locals():
                camera.release()

    # 执行五次循环，监听键盘输入
    for i in range(5):
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()  # 这是一个阻塞调用，直到监听器被手动停止

    # 清理GPIO设置
    arm.cleanup()
    relay_controller.cleanup()



if __name__ == "__main__":
    main8()