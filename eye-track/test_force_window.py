import cv2
import numpy as np
import ctypes

# Windows API constants
SW_RESTORE = 9

def main():
    window_name = "Eye Tracking Force Window Test"
    
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    # Create test image
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "WINDOW TEST", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(frame, "IF YOU SEE THIS, WINDOWS GUI IS WORKING", (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    cv2.imshow(window_name, frame)
    
    # Give OpenCV a moment to actually dispatch the window creation to the OS
    cv2.waitKey(100)
    
    # Use ctypes to find and force the window
    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, window_name)
    
    print(f"HWND: {hwnd}")
    if hwnd:
        print(f"IsWindow: {bool(user32.IsWindow(hwnd))}")
        print(f"IsWindowVisible: {bool(user32.IsWindowVisible(hwnd))}")
        
        user32.ShowWindow(hwnd, SW_RESTORE)
        user32.MoveWindow(hwnd, 100, 100, 900, 700, True)
        user32.SetForegroundWindow(hwnd)
        print("Forced window to foreground and moved to 100,100")
    else:
        print("Could not find HWND for the window!")
        
    print("Waiting 10 seconds. Look for the window!")
    cv2.waitKey(10000)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
