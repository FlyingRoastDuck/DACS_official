## Diversity-Authenticity Co-constrained Stylization for Federated Domain Generalization in Person Re-identification (AAAI'24)

### Introduction

This is the official repo for our AAAI 2024 paper "DACS".

![](figures/flowchart.pdf)

### Prerequisites

- CUDA>=11.7
- At least two TITAN X GPUs 
- Other necessary packages listed in [requirements.txt](requirements.txt)
- Download [ViT pre-trained model](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth) and put it under "./checkpoints"
- Training Data
  
  (Market-1501, DukeMTMC-reID and MSMT-17. You can download these datasets from [Zhong's repo](https://github.com/zhunzhong07/ECN))

   Unzip all datasets and ensure the file structure is as follow:
   
   ```
   DACS/data    
   │
   └───market1501 OR dukemtmc OR msmt17
        │   
        └───DukeMTMC-reID OR Market-1501-v15.09.15 OR MSMT17_V1
            │   
            └───bounding_box_train
            │   
            └───bounding_box_test
            | 
            └───query
            │   
            └───list_train.txt (only for MSMT-17)
            | 
            └───list_query.txt (only for MSMT-17)
            | 
            └───list_gallery.txt (only for MSMT-17)
            | 
            └───list_val.txt (only for MSMT-17)
   ```

### Usage

See [run.sh](run.sh) for details.

### Supplementary

The supplementary material of our paper is located at "figures/supp.pdf"

### How to Cite
```yang
@inproceedings{yang2024diversity,
  title={Diversity-Authenticity Co-constrained Stylization for Federated Domain Generalization in Person Re-identification},
  author={Yang, Fengxiang and Zhong, Zhun and Luo, Zhiming and He, Yifan and Li, Shaozi and Sebe, Nicu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={6},
  pages={6477--6485},
  year={2024}
}
```

### Contact Us

Email: yangfx@stu.xmu.edu.cn