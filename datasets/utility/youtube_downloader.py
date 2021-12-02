import os
import subprocess
import threading
from tqdm import tqdm
import shutil


def _read_outputs_from_precess(process):
    def _print_stdout(process):
        for line_ in iter(process.stdout.readline, ""):
            if len(line_) > 0:
                print(line_.strip())

    def _print_stderr(process):
        for line_ in iter(process.stderr.readline, ""):
            if len(line_) > 0:
                print(line_.strip())

    t1 = threading.Thread(target=_print_stdout, args=(process,))
    t2 = threading.Thread(target=_print_stderr, args=(process,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()


def download_youtube_videos(youtube_id_list, target_path: str, cache_path: str):
    for youtube_id in tqdm(youtube_id_list):
        youtube_video_path = os.path.join(target_path, youtube_id)
        if os.path.exists(youtube_video_path):
            continue
        url = f'https://www.youtube.com/watch?v={youtube_id}'
        # downloading_cache_path = os.path.join(cache_path, youtube_id)
        temp_path = os.path.join(target_path, f'{youtube_id}.tmp')
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        os.mkdir(temp_path)

        youtube_dl_output_path = os.path.join(temp_path, '%(title)s-%(id)s.%(ext)s')
        process = subprocess.Popen(['youtube-dl', '--cache-dir', cache_path, '-o', youtube_dl_output_path, url], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        _read_outputs_from_precess(process)

        process.wait()

        if process.returncode != 0:
            print(f'Failed to download video {youtube_id}')
            continue
        files = os.listdir(temp_path)
        if len(files) == 0:
            print(f'Youtube-dl returns 0, but nothing downloaded in video {youtube_id}')
            continue

        os.rename(temp_path, youtube_video_path)
