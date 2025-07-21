import subprocess
import RPi.GPIO as GPIO
from pynput import keyboard

# 全局变量，标记当前是否正在执行脚本
current_process = None


def cleanup_gpio():
    GPIO.cleanup()  # 清理所有GPIO设置
    print("GPIO resources have been released.")


def on_press(key):
    global current_process
    try:
        # 定义要执行的脚本与相应键的映射
        scripts = {
            '1': '/home/pi/AI-chess/01.py',
            '2': '/home/pi/AI-chess/02.py',
            '3': '/home/pi/AI-chess/03.py',
            '4': '/home/pi/AI-chess/04/04.py',
            '5': '/home/pi/AI-chess//05.py',
        }

        # 检查按下的键是否在映射中
        if key.char in scripts:
            if current_process and current_process.poll() is None:
                print("A script is still running. Please stop it first by pressing 't'.")
                return

            script_path = scripts[key.char]
            current_process = subprocess.Popen(['python3', script_path])  # 启动脚本
            print(f"Started executing {script_path}")

        elif key.char == 't':
            if current_process and current_process.poll() is None:
                current_process.terminate()  # 终止当前进程
                current_process.wait()  # 等待进程结束
                print("Current script has been terminated.")
                cleanup_gpio()  # 清理GPIO资源
                current_process = None
            else:
                print("No script is currently running.")

    except AttributeError:
        pass


def on_release(key):
    # 按下 'esc' 键退出监听
    if key == keyboard.Key.esc:
        if current_process and current_process.poll() is None:
            current_process.terminate()  # 终止当前进程
            current_process.wait()  # 等待进程结束
            print("Current script has been terminated.")
            cleanup_gpio()  # 清理GPIO资源
        return False


# 开始监听键盘事件
try:
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    cleanup_gpio()  # 在程序结束时清理GPIO资源

# 1 -5 分别代表运行 1 - 5 题目 t键盘代表终止程序
