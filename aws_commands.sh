git clone https://github.com/matteo-ronchetti/IKA.git
cd IKA
mkdir paper\ results/models
mkdir paper\ results/data
mkdir paper\ results/data/phototour
aws s3 cp s3://rosh-datasets/patches/notredame.pt paper\ results/data/phototour
aws s3 cp s3://rosh-datasets/patches/liberty.pt paper\ results/data/phototour
aws s3 cp s3://rosh-datasets/patches/yosemite.pt paper\ results/data/phototour
nvidia-docker run --rm -it -v$(pwd):/code -w /code deepmanager/gpu-pytorch bash

# in Docker
python setup.py install
pip install https://rosh-public.s3-eu-west-1.amazonaws.com/wheels/torch_batch_svd-0.0.0-cp37-cp37m-linux_x86_64.whl
cd paper\ results/
python evaluate_model.py
ls models/
wget -nc https://github.com/DagnyT/hardnet/raw/master/pretrained/train_liberty/checkpoint_liberty_no_aug.pth -P models/