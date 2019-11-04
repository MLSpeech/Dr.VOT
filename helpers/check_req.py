__author__ = 'YosiShrem'
try:
    import os
    import subprocess

    # praat
    if os.popen('uname -a').read().lower().__contains__("darwin"):
        assert os.path.exists("/Applications/Praat.app/Contents/MacOS/Praat"), \
            f"Couldn't Find praat at : '/Applications/Praat.app/Contents/MacOS/Praat'"
    elif os.popen('uname -a').read().lower().__contains__("linux"):
        assert os.path.exists(os.path.join(os.getcwd(), "linux_praaat")), \
            f"Couldn't find Praat at :{os.path.join(os.getcwd(), 'linux_praat')}"
    else:
        raise Exception("Unsupported Operating System, support only MacOS & Linux")
    # sox
    if not os.popen('sox --version').read().lower().__contains__("sox"):
        raise Exception(f"Couldn't find SoX. Check the tutorial for links")

    # pipenv
    if not os.popen('pipenv --version').read().lower().__contains__("pipenv"):
        raise Exception(f"Couldn't find pipenv. Check the tutorial for links")

except Exception as e:
    print(f"Error: {e}")
    exit(1)
