import torch
import os
import shutil
from miscellanies.torch.distributed import is_main_process


def _get_training_state_file_path(model_state_file_path: str):
    return os.path.join(os.path.dirname(model_state_file_path), os.path.splitext(os.path.basename(model_state_file_path))[0] + '-training.pth')


def _safe_rename(model_state_file_path, training_state_file_path, overwrite_existing=True):
    if overwrite_existing and os.path.exists(model_state_file_path):
        os.remove(model_state_file_path)
    if overwrite_existing and os.path.exists(training_state_file_path):
        os.remove(training_state_file_path)
    os.rename(model_state_file_path + '.tmp', model_state_file_path)
    os.rename(training_state_file_path + '.tmp', training_state_file_path)


def _fail_safe_save(path, model_state_dict, training_state_dict, overwrite_existing=True):
    training_state_file_path = _get_training_state_file_path(path)
    torch.save(model_state_dict, path + '.tmp')
    torch.save(training_state_dict, training_state_file_path + '.tmp')
    _safe_rename(path, training_state_file_path, overwrite_existing)


def _fail_safe_copy(src_path, dst_path, overwrite_existing=True):
    src_training_state_file_path = _get_training_state_file_path(src_path)
    dst_training_state_file_path = _get_training_state_file_path(dst_path)
    shutil.copy(src_path, dst_path + '.tmp')
    shutil.copy(src_training_state_file_path, dst_training_state_file_path + '.tmp')
    _safe_rename(dst_path, dst_training_state_file_path, overwrite_existing)


def dump_checkpoint(model_state_dict, objects_state_dict, epoch, output_path):
    if not is_main_process():
        return

    checkpoint_path = os.path.join(output_path, 'checkpoint.pth')
    _fail_safe_save(checkpoint_path, model_state_dict, objects_state_dict)
    backup_checkpoint_path = os.path.join(output_path, f'checkpoint{epoch:04}.pth')
    _fail_safe_copy(checkpoint_path, backup_checkpoint_path)


def load_checkpoint(checkpoint_path: str):
    training_state_file_path = _get_training_state_file_path(checkpoint_path)
    model_state_dict = torch.load(checkpoint_path, map_location='cpu')
    training_state_dict = torch.load(training_state_file_path, map_location='cpu')
    return model_state_dict, training_state_dict
