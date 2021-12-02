set -e
conda create -y -n SwinTrack
conda activate SwinTrack
conda install -y anaconda
conda install -y pytorch torchvision cudatoolkit -c pytorch
conda install -y -c fvcore -c iopath -c conda-forge fvcore
pip install wandb
pip install timm