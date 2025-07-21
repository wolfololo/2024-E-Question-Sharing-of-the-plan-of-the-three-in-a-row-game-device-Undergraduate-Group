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

            # 复制原始图像以绘制检测结果（此处不再需要显示）
            image_copy = self.image.copy()

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

            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=45)

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

        def draw_corners(self):
            if self.corner_coordinates:
                for (x, y) in self.corner_coordinates:
                    cv2.circle(self.image, (int(x), int(y)), 10, (0, 0, 255), -1)  # 红色圆点

                for i in range(4):
                    cv2.line(self.image, tuple(self.corner_coordinates[i]), tuple(self.corner_coordinates[(i + 1) % 4]), (0, 255, 0), 2)

                cv2.imshow('Corners and Pieces', self.image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("未能找到九宫格角点坐标信息。")

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
    detector.draw_corners()

    return board


# 测试
if __name__ == "__main__":
    main7('1.png')
