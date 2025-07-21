import RPi.GPIO as GPIO
import time
import cv2
import numpy as np
import copy







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
            1: (81, 60, 97) ,  # 位置9: 基座80°，肩部0°，肘部46° true1 1 1
            'A1': (75,64,63),  # 位置1: 基座97°，肩部14°，肘部0° 
            'B1':(104,68,49),  # 位置1: 基座97°，肩部14°，肘部0° true
            'C1':(106,66,60),  # 位置1: 基座97°，肩部14°，肘部0° ture
            'D1':(108, 70, 102),  # 位置1: 基座97°，肩部14°，肘部0°
            'A2':(75, 69,39),  # 位置1: 基座97°，肩部14°，肘部0° true
            'B2':(75, 64, 63),  # 位置1: 基座97°，肩部14°，肘部0° true
            'C2':(73, 70, 100),  # 位置1: 基座97°，肩部14°，肘部0°
            'D2':(70, 70, 112),  # 位置1: 基座97°，肩部14°，肘部0°
        }

        # 设置GPIO模式
        GPIO.setmode(GPIO.BCM)

        # 设置舵机引脚
        self.pwm = {}
        self.last_time = 0  # 上一次设置时间
        self.debounce_time = 0.5  # 防抖时间（秒）

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
            self.move_servo('base', angles[0])
            time.sleep(0.5)# 移动基座
            self.move_servo('elbow', angles[2])
            time.sleep(0.5)
            self.move_servo('shoulder', angles[1])  # 移动肩部
              # 移动肘部
        else:
            print("无效的位置编号！")
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




















def main1():
    # 初始化舵机控制
    arm_control = ArmControl()
    relay_controller = RelayController(relay_pin=17)  # 假设继电器连接在 GPIO 17 引脚

    try:
        # 初始位置（0）
        print("移动到初始位置")
        arm_control.remove_to_position(0)
        time.sleep(2)  # 延时2秒，确保机械臂到达初始位置

        # 移动到棋子A1位置
        print("移动到棋子A1位置")
        arm_control.move_to_position('B1')
        time.sleep(2.5)  # 延时2秒，确保机械臂到达棋子A1位置

        # 开启继电器并保持
        print("开启继电器")
        relay_controller.turn_on()
        time.sleep(1)  # 延时1秒，确保继电器开启并保持

        # 移动到位置5
        print("移动到位置5")
        arm_control.remove_to_position(5)
        time.sleep(2)  # 延时2秒，确保机械臂到达位置5

        # 关闭继电器
        print("关闭继电器")
        relay_controller.turn_off()
        time.sleep(5)  # 延时1秒，确保继电器关闭

        # 返回初始位置（0）
        print("返回初始位置")
        arm_control.remove_to_position(0)
        time.sleep(2)  # 延时2秒，确保机械臂返回初始位置

    except KeyboardInterrupt:
        print("操作中断")

    finally:
        # 清理GPIO设置
        arm_control.cleanup()
        relay_controller.cleanup()
        
        
        
        
        
if __name__ == "__main__":
    main1()
        
