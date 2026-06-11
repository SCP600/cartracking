from __future__ import annotations

from pathlib import Path
import sys


def _ensure_project_root_on_path() -> None:
    package_root = Path(__file__).resolve().parent
    project_root = package_root.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _set_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def main() -> None:
    _set_dpi_awareness()
    _ensure_project_root_on_path()

    from autocam_tracker.ui.main_window import MainWindow

    app = MainWindow()
    app.run()


if __name__ == "__main__":
    main()
