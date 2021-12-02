import subprocess
import os


def decode_video_file(video_file_path: str, destination_path: str, destination_file_name_pattern: str = '%06d',
                      destination_format: str = 'jpg'):
    subprocess.check_call(['ffmpeg', '-threads', '0', '-i', video_file_path,
                           os.path.join(destination_path, '{}.{}'.format(destination_file_name_pattern, destination_format))])
