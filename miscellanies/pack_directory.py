import tarfile
import os.path


def make_tarfile(output_filename, source_dir, compression_algorithm='xz'):
    with tarfile.open(output_filename, f'w:{compression_algorithm}') as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
