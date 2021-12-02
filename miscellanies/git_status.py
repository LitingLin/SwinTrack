import os
import subprocess
_cached_git_status = None


def _get_git_status():
    global _cached_git_status
    if _cached_git_status is not None:
        return _cached_git_status
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('utf-8').strip()
    sha = 'N/A'
    diff = 'clean'
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = 'dirty' if diff else 'clean'
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    git_state = sha, diff, branch
    _cached_git_status = git_state
    return git_state


def get_git_status_message():
    git_status = _get_git_status()
    git_diff = git_status[1]
    if git_diff == 'dirty':
        git_diff = 'has uncommited changes'
    message = f'sha: {git_status[0]}, diff: {git_diff}, branch: {git_status[2]}'
    return message


def get_git_status():
    return '-'.join(_get_git_status())
