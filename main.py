import os
import sys
import ctypes

if sys.platform == "win32":
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("FuzzyStride.App")

from ui.App import FuzzyStrideApp


def main():
    app = FuzzyStrideApp()
    app.run()


if __name__ == "__main__":
    main()