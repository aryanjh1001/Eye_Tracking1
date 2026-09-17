import ctypes
from dataclasses import dataclass

@dataclass
class ScreenInfo:
    width: int
    height: int
    dpi: int
    dpi_awareness: str
    conversion_required: bool

def init_dpi_and_get_screen() -> ScreenInfo:
    user32 = ctypes.windll.user32
    shcore = ctypes.windll.shcore

    dpi_awareness_str = "Unknown"
    # Try SetProcessDpiAwarenessContext (Windows 10 1607+)
    try:
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            dpi_awareness_str = "PerMonitorV2"
    except AttributeError:
        # Fallback to SetProcessDpiAwareness (Windows 8.1+)
        try:
            shcore.SetProcessDpiAwareness(2)
            dpi_awareness_str = "PerMonitor (fallback)"
        except AttributeError:
            user32.SetProcessDPIAware()
            dpi_awareness_str = "SystemAware (legacy)"

    width = user32.GetSystemMetrics(0)  # SM_CXSCREEN
    height = user32.GetSystemMetrics(1) # SM_CYSCREEN

    # Get monitor DPI
    # MonitorFromWindow(None, MONITOR_DEFAULTTOPRIMARY=1)
    hmonitor = user32.MonitorFromWindow(0, 1)
    dpi_x = ctypes.c_uint()
    dpi_y = ctypes.c_uint()
    try:
        # MDT_EFFECTIVE_DPI = 0
        shcore.GetDpiForMonitor(hmonitor, 0, ctypes.byref(dpi_x), ctypes.byref(dpi_y))
        dpi = dpi_x.value
    except AttributeError:
        # GetDeviceCaps fallback
        hdc = ctypes.windll.gdi32.CreateDCW("DISPLAY", None, None, None)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88) # LOGPIXELSX
        ctypes.windll.gdi32.DeleteDC(hdc)

    # If DPI awareness worked, the reported size is the physical size.
    # We will assume coordinate conversion is not required, as long as OpenCV windows match this.
    conversion_required = False

    return ScreenInfo(
        width=width,
        height=height,
        dpi=dpi,
        dpi_awareness=dpi_awareness_str,
        conversion_required=conversion_required
    )
