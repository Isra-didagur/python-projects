from pyautogui import *
import pyautogui
import time
import keyboard
import random
import win32api, win32con


time.sleep(1)
jumps = 0

def jump():
    global jumps
    win32api.keybd_event(38, 0, 0, 0)
    time.sleep(0.01)
    win32api.keybd_event(38, 0, win32con.KEYEVENTF_KEYUP, 0)
    jumps += 1


def duck():
    win32api.keybd_event(40, 0, 0, 0)
    time.sleep(0.4)
    win32api.keybd_event(40, 0, win32con.KEYEVENTF_KEYUP, 0)
    print("I see a Pterodactyl")


# Press 'q' to stop running
print("Script is running. Press 'q' to quit.")
while not keyboard.is_pressed('q'):
    # Check the color of the first pixel
    tall_cactus_color = pyautogui.pixel(730, 468)
    print(f"Tall cactus check at (730, 468): {tall_cactus_color}")

    # Check the color of the second pixel
    short_cactus_color = pyautogui.pixel(715, 490)
    print(f"Short cactus check at (715, 490): {short_cactus_color}")

    # Check the color of the third pixel
    pterodactyl_color = pyautogui.pixel(760, 460)
    print(f"Pterodactyl check at (760, 460): {pterodactyl_color}")

    # Your original logic
    if tall_cactus_color[0] == 83:
        jump()
        print("I see a tall cactus!")
    elif short_cactus_color[0] == 83:
        jump()
        print("I see a short cactus!")
    
    if pterodactyl_color[0] == 83:
        duck()
    
    if jumps > 3:
        # Check for speed increases
        faster_tall_cactus_color = pyautogui.pixel(720, 468)
        print(f"Faster tall cactus check at (720, 468): {faster_tall_cactus_color}")
        
        faster_short_cactus_color = pyautogui.pixel(705, 490)
        print(f"Faster short cactus check at (705, 490): {faster_short_cactus_color}")
        
        faster_pterodactyl_color = pyautogui.pixel(750, 460)
        print(f"Faster pterodactyl check at (750, 460): {faster_pterodactyl_color}")
        
        if faster_tall_cactus_color[0] == 83:
            jump()
            print("I see a tall cactus!")
        elif faster_short_cactus_color[0] == 83:
            jump()
        if faster_pterodactyl_color[0] == 83:
            duck()