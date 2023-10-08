# LOANet: A Lightweight Network Using Object Attention for Extracting Buildings and Roads from UAV Aerial Remote Sensing Images
by [Xiaoxiang Han](https://linyer.ac.cn/), [Yiman Liu](https://linyer.ac.cn/authors/yiman-liu/), [Gang Liu](http://www.eqhb.gov.cn/info/1143/21018.htm), Yuanjie Lin, [Qiaohong Liu](https://yjsjy.sumhs.edu.cn/lqh/main.htm).

## Introduction
Official code for "[LOANet: A Lightweight Network Using Object Attention for Extracting Buildings and Roads from UAV Aerial Remote Sensing Images](https://doi.org/10.7717/peerj-cs.1467)".

## Usage
Requirements are in `requirements.txt`.
Hyper-parameters are in `config.yaml`.

To train a model,
```
python train.py fit -c config.yaml  #for LOANet
python train.py fit -c config_lg.yaml  #for LOANet_large
``` 

To test a model,
```
python test.py
``` 

## Citation

If you find these projects useful, please consider citing:

```bibtex
@article{han2023loanet,
  title={LOANet: a lightweight network using object attention for extracting buildings and roads from UAV aerial remote sensing images},
  author={Han, Xiaoxiang and Liu, Yiman and Liu, Gang and Lin, Yuanjie and Liu, Qiaohong},
  journal={PeerJ Computer Science},
  volume={9},
  pages={e1467},
  year={2023},
  publisher={PeerJ Inc.}
}
```