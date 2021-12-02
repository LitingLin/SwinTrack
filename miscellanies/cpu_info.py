import os, platform, subprocess, re


def get_processor_name():
    if platform.system() == "Windows":
        name = subprocess.check_output(["wmic", "cpu", "get", "name"], universal_newlines=True).strip().split("\n")[-1]
        return name
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command = "sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True, universal_newlines=True).strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub( ".*model name.*:", "", line,1)
    return ""
