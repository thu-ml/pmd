Installation
====

    git clone git@github.com:cjf00000/pmd.git
    cd pmd
    pip install -e .

Generative Model Usage
====
The scripts will automatically download MNIST and CIFAR10 datasets. The LFW dataset and SVHN dataset can currently be found in jungpu10:/home/jianfei/pmd-data.

Running:
   
    cd generative-model
    mkdir data
    CUDA_VISIBLE_DEVICES=0 configs/mnist_pmd_fc
