git clone https://github.com/matteo-ronchetti/IKA.git
cd IKA
mkdir paper\ results/models
mkdir paper\ results/data
mkdir paper\ results/data/phototour
pip install awscli
aws s3 cp s3://rosh-datasets/patches/notredame.pt paper\ results/data/phototour
aws s3 cp s3://rosh-datasets/patches/liberty.pt paper\ results/data/phototour
aws s3 cp s3://rosh-datasets/patches/yosemite.pt paper\ results/data/phototour
aws s3 cp s3://rosh-datasets/patches/model_4096.pth paper\ results/models

python setup.py install
pip install https://rosh-public.s3-eu-west-1.amazonaws.com/wheels/torch_batch_svd-0.0.0-cp37-cp37m-linux_x86_64.whl
cd paper\ results/
wget -nc https://github.com/DagnyT/hardnet/raw/master/pretrained/train_liberty/checkpoint_liberty_no_aug.pth -P models/


#aws s3 cp s3://rosh-datasets/patches/precomputed_yos_02_20k_30.npz paper\ results
nvidia-docker run --rm -it -v$(pwd):/code -w /code deepmanager/gpu-pytorch bash

# in Docker
python setup.py install
pip install https://rosh-public.s3-eu-west-1.amazonaws.com/wheels/torch_batch_svd-0.0.0-cp37-cp37m-linux_x86_64.whl
pip install dlib
conda install faiss-gpu -y -c pytorch
cd paper\ results/
wget -nc https://github.com/DagnyT/hardnet/raw/master/pretrained/train_liberty/checkpoint_liberty_no_aug.pth -P models/


#python integration.py --sampling-points 50 --sigma 0.5 --output integrated_5_25.npy
#python train_invariant_kernel.py --factor integrated_5_25.npy --output model_5_25.pth
#python evaluate_model.py --dataset notredame --model hardnet+ika --model-path model_5_25.pth
#python evaluate_model.py --dataset yosemite --model hardnet+ika --model-path model_5_25.pth
