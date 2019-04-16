import platform

separator = "/"
if 'Windows' in platform.system():
    separator = "\\"
else:
    separator = "/"