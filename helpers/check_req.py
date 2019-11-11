__author__ = 'YosiShrem'
try:
    import os
    import subprocess
    import sys
    if not (sys.version_info[0]>=3 and sys.version_info[1]>=6):
        raise SystemExit("[ERROR] support Python 3.6+, your version is {}".format('.'.join([str(i) for i in sys.version_info])))
    # praat
    if os.popen('uname -a').read().lower().__contains__("darwin"):
        if not os.path.exists("/Applications/Praat.app"):
            raise SystemExit("Couldn't Find praat at : '/Applications/Praat.app', please make sure praat exists at this location")
        if not os.path.exists("/Applications/Praat.app/Contents/MacOS/Praat"):
            raise SystemExit("praat.app is corrupted, please re-install")

    elif os.popen('uname -a').read().lower().__contains__("linux"):
        if not os.path.exists(os.path.join(os.getcwd(), "linux_praat")):
            raise SystemExit("Couldn\'t find Praat at :{}".format(os.path.join(os.getcwd(), 'linux_praat')))
    else:
        raise SystemExit("Unsupported Operating System, support only MacOS & Linux")
    # sox
    if not os.popen('sox --version').read().lower().__contains__("sox"):
        raise SystemExit(f"Couldn't find SoX. Check the tutorial for links")

    # pipenv
    if not os.popen('pipenv --version').read().lower().__contains__("pipenv"):
        raise SystemExit(f"Couldn't find pipenv. Check the tutorial for links")

except Exception as e:
    print(f"Error: {e}")
    exit(1)
