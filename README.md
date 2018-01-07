Population Matching Discrepancy
====

Code for the paper [Population Matching Discrepancy and Applications in Deep Learning](https://papers.nips.cc/paper/7206-population-matching-discrepancy-and-applications-in-deep-learning.pdf).


Installation
----

Make sure that you have valid c++11 and cuda compilers.

    git clone git@github.com:cjf00000/pmd.git
    cd pmd
    pip install --upgrade pip
    pip install cython numpy tensorflow scipy matplotlib scikit-image scikit-learn seaborn
    pip install -e .


Domain Adaptation Usage
----

Download the data

    cd domain_adaptation
    wget -O data.tar.gz https://github.com/cjf00000/pmd-data/blob/master/data.tar.gz?raw=true
    tar xzvf data.tar.gz

See `config.pmd` or `config.mmd` for the training recipes.


Generative Model Usage
----
The scripts will automatically download MNIST and CIFAR10 datasets.

Running:
   
    cd generative-model
    mkdir data
    configs/mnist_pmd_fc

The images will be generated in the `result` directory.


Cite
----

If you find the code is useful, please cite our paper!

    @inproceedings{chen2017population,
      title={Population Matching Discrepancy and Applications in Deep Learning},
      author={Chen, Jianfei and Chongxuan, LI and Ru, Yizhong and Zhu, Jun},
      booktitle={Advances in Neural Information Processing Systems},
      pages={6263--6275},
      year={2017}
    }
