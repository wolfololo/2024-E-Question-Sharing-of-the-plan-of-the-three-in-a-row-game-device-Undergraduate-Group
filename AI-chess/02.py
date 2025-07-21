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


def on_release(key):
    if key == keyboard.Key.esc:
        # 退出程序的条件
        return False  # 停止监听



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
            0:(60 ,60 , 90), 
            9: (94, 69, 46),  # 位置1: 基座97°，肩部14°，肘部0° true1 9 1
            8: (89, 67, 48),  # 位置2: 基座90°，肩部14°，肘部0° true1 8 1
            7: (83, 63, 51),  # 位置3: 基座82°，肩部14°，肘部0° true1 7 1
            6: (96,61, 67),   # 位置4: 基座97°，肩部5°，肘部3°  true1 6 1
            5: (89, 61, 73),   # 位置5: 基座90°，肩部0°，肘部0° true1 5 1
            4: (82, 61, 74),   # 位置6: 基座82°，肩部5°，肘部3° true1 4 1
            3: (98, 60, 89),  # 位置7: 基座99°，肩部0°，肘部45° true1 3 1
            2: (89, 60, 93),  # 位置8: 基座90°，肩部0°，肘部45°  true1 2 1
            1: (80, 60, 97) ,  # 位置9: 基座80°，肩部0°，肘部46° true1 1 1
            'A1': (75,64,63),  # 位置1: 基座97°，肩部14°，肘部0° 
            'B1':(104,67,49),  # 位置1: 基座97°，肩部14°，肘部0° true
            'C1':(106,66,67),  # 位置1: 基座97°，肩部14°，肘部0° ture
            'D1':(108, 70, 102),  # 位置1: 基座97°，肩部14°，肘部0°
            'A2':(75, 69,43),  # 位置1: 基座97°，肩部14°，肘部0° true 1
            'B2':(75, 64, 67),  # 位置1: 基座97°，肩部14°，肘部0° true
            'C2':(73, 64, 85),  # 位置1: 基座97°，肩部14°，肘部0°
            'D2':(70, 70, 112),  # 位置1: 基座97°，肩部14°，肘部0°
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
            self.move_servo('base', angles[0])# 移动基座
            time.sleep(0.5)
            self.move_servo('elbow', angles[2])
            time.sleep(1)
            self.move_servo('shoulder', angles[1])
            # 移动肩部
              # 移动肘部
    def remove_to_position(self, position):
        """移动机械臂到指定棋盘位置（1-9）"""
        if position in self.position_angles:
            angles = self.position_angles[position]  # 获取角度元组
            self.move_servo('shoulder', angles[1])# 移动基座
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
            11: (104, 68, 49),
            12: (75, 64, 67),
            13: (106, 66, 67),
            14: (73, 64, 85)
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


# 初始化一个计数器变量来记录 '1' 键的按下次数
key1_press_count = 0

def on_press(key):
    global relay_controller, key1_press_count
    relay_controller = RelayController(relay_pin=17)
    try:
        # 替换按下 '1' 键来执行相应的功能
        if key.char == '1':
            arm_control_a.move_to_next_sequence_position()
            print("yidongwanc")

        # 如果按下 'q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c'
        if key.char in 'qweasdzxc':
            position = 'qweasdzxc'.index(key.char) + 1  # 将字符转换为对应的数字 1-9
            relay_controller.turn_on()q1w1e1a
            time.sleep(2)
            arm_control_a.remove_to_position(position)
            time.sleep(2)
            relay_controller.turn_off()

    except AttributeError:
        # 如果按下的是功能键（如 Shift、Ctrl 等），则不做处理
        relay_controller.cleanup()
        pass



def main6():
    global arm_control_a
    arm_control_a = ArmControlA()
      # 示例继电器 GPIO 引脚

    print("按 'i' 键依次移动到序列位置 11-14。按 'qweasdzxc' 键移动到 ArmControl 的位置。")

    # 启动键盘监听
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            listener.join()  # 开始监听，直到 Esc 键被按下
        except KeyboardInterrupt:
            print("程序终止。清理 GPIO 设置。")
            arm_control_a.cleanup()
            relay_controller.cleanup()



# 1
# q w e a s d z x c





if __name__ == "__main__":
    main6()