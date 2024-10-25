# feature-preserve-portrait-editing

This is the code for Learning Feature-Preserving Portrait Editing from Generated Pairs

 * [Arxiv](https://arxiv.org/abs/2407.20455)


## Setup
We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to set up an environment:

    conda create --name mcdm python=3.8.5

    conda activate mcdm

Install the required packages:

    pip install -r requirements.txt 



## Demo on a Single Image 

First, download pretrained model weights of two tasks (outfit editing and cartoon expression editing) from [HuggingFace](https://huggingface.co/ByteDance/feature-preserve-portrait-editing) and put them into the root folder:

We provide examples for the outfit editing task. To apply all four supported outfit editing effects, simply run the following command. The results will be saved in `./demo_results`. 

    bash demo.sh 

Alternatively, you can also run following commands for a specific editing like below:

    python demo.py --model_dir portrait_editing_models/outfit/checkpoint-200000 --image_path ./data/outfit/test/image1.jpg --prompt "a man, cute flower costume"   




## Model Training

We have provided a small sample dataset for outfit editing, located in the `./data directory`. This dataset is intended for reference purposes.

To train the model with your own dataset, ensure that your dataset follows the same structure as the provided sample dataset.

To start training, run the following command:

    bash train.sh



## Acknowledgement

This codebase is adpated from [diffusers](https://github.com/huggingface/diffusers) and [DreamPose](https://github.com/johannakarras/DreamPose).


## License
```
Copyright 2024 Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Citation

    @article{chen2024learning,
    title={Learning Feature-Preserving Portrait Editing from Generated Pairs},
    author={Chen, Bowei and Zhi, Tiancheng and Zhu, Peihao and Sang, Shen and Liu, Jing and Luo, Linjie},
    journal={arXiv preprint arXiv:2407.20455},
    year={2024}
    }

