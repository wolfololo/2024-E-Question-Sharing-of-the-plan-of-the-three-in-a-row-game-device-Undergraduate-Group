import cv2
import numpy as np

def main7():
    class BHSBFL:
        def __init__(self, image):
            self.image = image
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

                coordinates = []
                for i in range(3):
                    for j in range(3):
                        center_x = int(box[0][0] + j * cell_width + cell_width / 2)
                        center_y = int(box[0][1] + i * cell_height + cell_height / 2)

                        coordinates.append((i * 3 + j + 1, (center_x, center_y)))

                return coordinates
            else:
                print("未找到轮廓")
                return []

    class ChessPieceDetector(BHSBFL):
        def __init__(self, image):
            super().__init__(image)
            self.coordinates = self.find_grid_coordinates()
            self.corner_coordinates = self.find_corners()

        def find_corners(self):
            edged = cv2.Canny(self.gray, 30, 150)
            contours_info = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(largest_contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                return box.tolist()
            return []

        def non_max_suppression(self, boxes, overlapThresh=0.3):
            if len(boxes) == 0:
                return []

            boxes = np.array(boxes)
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)

            idxs = np.argsort(y2)

            pick = []

            while len(idxs) > 0:
                last = len(idxs) - 1
                i = idxs[last]
                pick.append(i)

                xx1 = np.maximum(x1[i], x1[idxs[:last]])
                yy1 = np.maximum(y1[i], y1[idxs[:last]])
                xx2 = np.minimum(x2[i], x2[idxs[:last]])
                yy2 = np.minimum(y2[i], y2[idxs[:last]])

                w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)
                overlap = (w * h) / area[idxs[:last]]

                idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

            return boxes[pick].astype("int")

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

                boxes = np.array([[x - r, y - r, x + r, y + r] for (x, y, r) in circles])
                scores = np.array([1] * len(circles))

                # 使用自定义的非极大值抑制
                boxes = boxes.tolist()
                nms_boxes = self.non_max_suppression(boxes, overlapThresh=0.5)

                for box in nms_boxes:
                    x = (box[0] + box[2]) // 2
                    y = (box[1] + box[3]) // 2
                    r = (box[2] - box[0]) // 2
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
                        point_color = (0, 0, 255) if piece_color == "黑色" else (0, 255, 0)
                        cv2.circle(self.image, (x, y), 10, point_color, -1)

            piece_positions.sort(key=lambda x: x[0])

            return piece_positions

        def draw_corners(self):
            if self.corner_coordinates:
                for (x, y) in self.corner_coordinates:
                    cv2.circle(self.image, (int(x), int(y)), 10, (0, 0, 255), -1)

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

    def resize_image(image, scale_percent):
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        if not ret:
            print("无法读取摄像头图像")
            break

        detector = ChessPieceDetector(frame)
        piece_positions = detector.detect_pieces()

        for number, color in piece_positions:
            print(f"棋子位置编号: {number}, 颜色: {color}")

        # 将图像缩小到 50%
        resized_frame = resize_image(frame, 50)
        cv2.imshow('Camera Feed', resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main7()
