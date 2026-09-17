import ctypes
import ctypes.wintypes
import time

# Windows constants
WS_OVERLAPPEDWINDOW = 0x00CF0000
WS_VISIBLE = 0x10000000
CS_HREDRAW = 0x0002
CS_VREDRAW = 0x0001
SW_SHOW = 5
WM_DESTROY = 0x0002
WM_PAINT = 0x000F
WM_QUIT = 0x0012
PM_REMOVE = 0x0001
DT_CENTER = 0x0001
DT_SINGLELINE = 0x0020
HWND_TOPMOST = -1
HWND_NOTOPMOST = -2
SWP_SHOWWINDOW = 0x0040

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32

# Properly declare DefWindowProcW for 64-bit
user32.DefWindowProcW.argtypes = [
    ctypes.wintypes.HWND,
    ctypes.c_uint,
    ctypes.wintypes.WPARAM,
    ctypes.wintypes.LPARAM,
]
user32.DefWindowProcW.restype = ctypes.c_longlong

# WNDPROC must use pointer-sized integers for WPARAM/LPARAM
WNDPROC = ctypes.WINFUNCTYPE(
    ctypes.c_longlong,
    ctypes.wintypes.HWND,
    ctypes.c_uint,
    ctypes.wintypes.WPARAM,
    ctypes.wintypes.LPARAM,
)

class WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", ctypes.c_uint),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", ctypes.wintypes.HINSTANCE),
        ("hIcon", ctypes.wintypes.HICON),
        ("hCursor", ctypes.wintypes.HICON),
        ("hbrBackground", ctypes.wintypes.HBRUSH),
        ("lpszMenuName", ctypes.wintypes.LPCWSTR),
        ("lpszClassName", ctypes.wintypes.LPCWSTR),
    ]

class PAINTSTRUCT(ctypes.Structure):
    _fields_ = [
        ("hdc", ctypes.wintypes.HDC),
        ("fErase", ctypes.wintypes.BOOL),
        ("rcPaint", ctypes.wintypes.RECT),
        ("fRestore", ctypes.wintypes.BOOL),
        ("fIncUpdate", ctypes.wintypes.BOOL),
        ("rgbReserved", ctypes.c_byte * 32),
    ]


def wnd_proc(hwnd, msg, wparam, lparam):
    if msg == WM_PAINT:
        ps = PAINTSTRUCT()
        hdc = user32.BeginPaint(hwnd, ctypes.byref(ps))

        # Fill with bright blue
        brush = gdi32.CreateSolidBrush(0x00FF4040)
        rect = ctypes.wintypes.RECT()
        user32.GetClientRect(hwnd, ctypes.byref(rect))
        user32.FillRect(hdc, ctypes.byref(rect), brush)
        gdi32.DeleteObject(brush)

        gdi32.SetBkMode(hdc, 1)  # TRANSPARENT
        gdi32.SetTextColor(hdc, 0x0000FFFF)  # yellow

        font = gdi32.CreateFontW(48, 0, 0, 0, 700, 0, 0, 0, 0, 0, 0, 0, 0, "Arial")
        old_font = gdi32.SelectObject(hdc, font)

        r1 = ctypes.wintypes.RECT(0, 80, rect.right, 150)
        text1 = "NATIVE WINDOWS TEST"
        user32.DrawTextW(hdc, text1, len(text1), ctypes.byref(r1), DT_CENTER | DT_SINGLELINE)

        gdi32.SetTextColor(hdc, 0x00FFFFFF)  # white
        font2 = gdi32.CreateFontW(30, 0, 0, 0, 400, 0, 0, 0, 0, 0, 0, 0, 0, "Arial")
        gdi32.SelectObject(hdc, font2)
        r2 = ctypes.wintypes.RECT(0, 200, rect.right, 260)
        text2 = "IF YOU SEE THIS, WINDOWS GUI IS VISIBLE"
        user32.DrawTextW(hdc, text2, len(text2), ctypes.byref(r2), DT_CENTER | DT_SINGLELINE)

        gdi32.SelectObject(hdc, old_font)
        gdi32.DeleteObject(font)
        gdi32.DeleteObject(font2)

        user32.EndPaint(hwnd, ctypes.byref(ps))
        return 0

    elif msg == WM_DESTROY:
        user32.PostQuitMessage(0)
        return 0

    return user32.DefWindowProcW(hwnd, msg, wparam, lparam)


def main():
    hInstance = kernel32.GetModuleHandleW(None)
    class_name = "NativeTestWindowClass2"
    title = "NATIVE WINDOWS VISIBILITY TEST"

    wnd_proc_cb = WNDPROC(wnd_proc)

    wc = WNDCLASSW()
    wc.style = CS_HREDRAW | CS_VREDRAW
    wc.lpfnWndProc = wnd_proc_cb
    wc.hInstance = hInstance
    wc.hCursor = user32.LoadCursorW(0, 32512)
    wc.hbrBackground = gdi32.CreateSolidBrush(0x00FF4040)
    wc.lpszClassName = class_name

    atom = user32.RegisterClassW(ctypes.byref(wc))
    if not atom:
        print(f"RegisterClassW FAILED: {kernel32.GetLastError()}")
        return
    print(f"RegisterClassW: atom={atom}")

    hwnd = user32.CreateWindowExW(
        0, class_name, title,
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        100, 100, 800, 500,
        0, 0, hInstance, 0,
    )
    if not hwnd:
        print(f"CreateWindowExW FAILED: {kernel32.GetLastError()}")
        return

    print(f"HWND: {hwnd}")
    print(f"IsWindow: {bool(user32.IsWindow(hwnd))}")
    print(f"IsWindowVisible: {bool(user32.IsWindowVisible(hwnd))}")

    rect = ctypes.wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    print(f"GetWindowRect: left={rect.left}, top={rect.top}, right={rect.right}, bottom={rect.bottom}")

    user32.ShowWindow(hwnd, SW_SHOW)
    user32.UpdateWindow(hwnd)
    user32.SetWindowPos(hwnd, HWND_TOPMOST, 100, 100, 800, 500, SWP_SHOWWINDOW)
    user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 100, 100, 800, 500, SWP_SHOWWINDOW)
    user32.SetForegroundWindow(hwnd)

    rect2 = ctypes.wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect2))
    print(f"GetWindowRect AFTER: left={rect2.left}, top={rect2.top}, right={rect2.right}, bottom={rect2.bottom}")

    print()
    print("Window should be visible NOW. Look at your screen!")
    print("Keeping alive for 15 seconds...")

    start = time.time()
    msg = ctypes.wintypes.MSG()
    while time.time() - start < 15.0:
        while user32.PeekMessageW(ctypes.byref(msg), 0, 0, 0, PM_REMOVE):
            if msg.message == WM_QUIT:
                print("WM_QUIT received.")
                return
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        time.sleep(0.01)

    print("15 seconds elapsed. Destroying window.")
    user32.DestroyWindow(hwnd)
    print("Done.")


if __name__ == "__main__":
    main()
