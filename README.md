# dark_infer_collections
Here are the low-light image enhancement inference code collections. 

## Why this project?
Thanks to the researchers for contributing to facilitating the low-light image enhancement field. Nevertheless, I find that there are still many inconveniences for followers due to various factors:

* **Framework**: Some of the models were implemented by Tensorflow, and some were by Pytorch. On the other hand, early works used old-version frameworks, which is not suitable for the recent ones.

* **Coding style**: Different coders have different coding styles. It occasionally makes the code reading difficult.

* **Closed source**: Some researchers have not released their codes publicly. For newbies, reproducing codes is hard.

Therefore, I want to create a unified codes repository for low-light image enhancement by the latest PyTorch framework. This repository includes the inference code collections. If I have more time, I will organize training codes later.
Welcome to join the project!


## Recent updates
#### 2022.8.3 add KinD++ (by ME)
#### 2022.8.3 add KinD (by ME)
#### 2022.8.3 add UTVNet (by ME)
#### 2022.7.30 add Gamma (by ME)
#### 2022.7.28 add SCI (by ME)
#### 2022.5.2 add RUAS
#### 2022.4.30 add DSLR
#### 2022.4.29 add ReLLIE (**need to improve**)
#### 2022.4.19 add DALE
#### 2022.4.17 add DRBN(stage1, stage2) & SGM
#### 2022.4.16 add EnlightenGAN
#### 2022.4.14 add ZeroDCE & ZeroDCE++
#### 2022.4.12 start project

## Requirements
install 

    python 3.6+
    pillow
    torch
    torchvision

or 

    pip install -r requirements.txt

## How to run
Download the pre-trained models([OneDrive Password:ymshi](https://mailustceducn-my.sharepoint.com/:f:/g/personal/ymshi_mail_ustc_edu_cn/Ejoo9mGJzihDoHRKKB6TL9MBO6G_jAy5nqejHa-jWoprpw)) and put "weights" folder into the  "dark_infer_collections" folder.

If testing only one model, e.g., ZeroDCE, you can run the code as follows:

    python ZeroDCE/infer.py

Then you can see the results in the folder "output".

<!-- You can also modify the `in_path` and `out_path` to your own. -->

## TODO

<!-- ✅ ⭕️ ❌-->
|Pytorch|Tensorflow/Keras|Other framework|Non-public|
|---  |---  |--- | --- |
|✅ DRBN |⭕️ Retinex-Net |⭕️ SICE|⭕️ D&E| 
|⭕️ DRBN-v2 |⭕️ GladNet |⭕️ LightenNet |⭕️ DLN |
|✅ SGM |⭕️ DeepUPE |⭕️ LLNet | ⭕️ PRIEN|
|✅ EnlightenGAN |✅ KinD | ⭕️TBEFN|  ⭕️ ProRetinex|
|✅ ZeroDCE |✅ KinD++ |⭕️ExCNet | ⭕️Component-GAN|
|✅ ZeroDCE++ |⭕️ ISSR  | | |
|✅  DALE |⭕️ MBLLEN  | | |
|✅ DSLR |⭕️ AGLLNet | | |
|⭕️ StableLLVE| | | |
|⭕️ LPNet| | | |
|✅ ReLLIE| | | |
|✅ RUAS| | | |
|⭕️ RRDNet| | | |

# License
The codes are made available for academic research purpose only.
## Other links
[Lighting-the-Darkness-in-the-Deep-Learning-Era-Open](https://github.com/Li-Chongyi/Lighting-the-Darkness-in-the-Deep-Learning-Era-Open)


