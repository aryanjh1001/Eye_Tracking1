import cv2
import numpy as np
import ctypes
import ctypes.wintypes

TITLE = "EYE_TRACKING_VISIBILITY_TEST_2026"

def main():
    # Create test image with large text
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)  # dark gray background
    cv2.putText(img, "EYE TRACKING", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
    cv2.putText(img, "WINDOW TEST", (80, 250), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)
    cv2.putText(img, "YOU SHOULD SEE THIS", (30, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # Create and configure window
    cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TITLE, 640, 480)
    cv2.moveWindow(TITLE, 100, 100)

    # Show image
    cv2.imshow(TITLE, img)

    # Pump message queue so window is fully realized
    cv2.waitKey(100)

    # ctypes user32
    user32 = ctypes.windll.user32

    hwnd = user32.FindWindowW(None, TITLE)
    print(f"HWND: {hwnd}")
    print(f"IsWindow: {bool(user32.IsWindow(hwnd))}")
    print(f"IsWindowVisible: {bool(user32.IsWindowVisible(hwnd))}")

    rect = ctypes.wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    print(f"GetWindowRect BEFORE: left={rect.left}, top={rect.top}, right={rect.right}, bottom={rect.bottom}")

    # Force to foreground
    SW_RESTORE = 9
    HWND_TOPMOST = -1
    HWND_NOTOPMOST = -2
    SWP_SHOWWINDOW = 0x0040

    user32.ShowWindow(hwnd, SW_RESTORE)
    user32.SetWindowPos(hwnd, HWND_TOPMOST, 100, 100, 640, 480, SWP_SHOWWINDOW)
    user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 100, 100, 640, 480, SWP_SHOWWINDOW)
    user32.SetForegroundWindow(hwnd)

    rect2 = ctypes.wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect2))
    print(f"GetWindowRect AFTER:  left={rect2.left}, top={rect2.top}, right={rect2.right}, bottom={rect2.bottom}")

    print()
    print("Window should now be visible at (100, 100).")
    print("Keeping window alive for 15 seconds. LOOK AT YOUR SCREEN NOW.")

    cv2.waitKey(15000)

    cv2.destroyAllWindows()
    print("Done. Window closed.")

if __name__ == "__main__":
    main()
