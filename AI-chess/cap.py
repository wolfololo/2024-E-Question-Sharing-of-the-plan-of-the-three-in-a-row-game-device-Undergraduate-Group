import cv2
from pynput import keyboard

def take_photo(filename='5.png'):
    # 打开默认摄像头
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 读取一帧
    ret, frame = cap.read()

    if ret:
        # 保存图像
        cv2.imwrite(filename, frame)
        print(f"照片已保存为 {filename}")
    else:
        print("无法读取图像")

    # 释放摄像头
    cap.release()

def on_press(key):
    try:
        if key.char == 'a':
            take_photo()  # 按下 'a' 键时调用拍照函数
        elif key.char == 'q':
            print("退出程序")
            return False  # 返回 False 将停止监听
    except AttributeError:
        # 处理特殊键（如 Shift、Ctrl 等），不做处理
        pass

def main():
    print("按 'a' 键拍照，按 'q' 键退出程序")

    # 监听键盘按键
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()  # 等待监听结束

if __name__ == "__main__":
    main()
