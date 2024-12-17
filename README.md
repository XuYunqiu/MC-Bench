# MC-Bench

[![Website](https://img.shields.io/badge/Project-Website-F9D371)](https://xuyunqiu.github.io/MC-Bench/)
[![paper](https://img.shields.io/badge/arXiv-Paper-blue.svg)](https://arxiv.org/abs/2410.12332)
[![Dataset](https://img.shields.io/badge/Dataset-Access-<F9D371)](https://drive.google.com/drive/folders/1EWEiECUcJn4YD91z_n8FfHS1z5VKFi16?usp=drive_link)

This repo contains data and evaluation code for MC-Bench, a new benchmark for multi-context visual grounding.


> **Abstract:** *While multimodal large language models (MLLMs) have demonstrated extraordinary vision-language understanding capabilities, their abilities to solve instance-level visual-language problems beyond a single image warrant further exploration. To assess these unproven abilities of MLLMs, this paper proposes a new visual grounding task called multi-context visual grounding, which aims to localize instances of interest across multiple images based on open-ended text prompts. In order to facilitate this research, we construct a new dataset MC-Bench that features 2K high-quality and manually annotated samples. Each sample consists of an instance-level labeled image pair and a corresponding text prompt that indicates the target instances in the images. These text prompts are highly open-ended and grouped into three distinct styles, covering 20 practical skills. We benchmark over 20 state-of-the-art MLLMs and foundation models with potential multi-context visual grounding capabilities, as well as a simple yet effective stepwise baseline and a finetuned baseline by multi-context instruction tuning. Our evaluation reveals a non-trivial performance gap between existing MLLMs and humans, along with some interesting observations that suggest potential future directions. We hope MC-Bench and our empirical findings can encourage the research community to further explore and enhance the untapped potentials of MLLMs in instance-level tasks, particularly in multi-image contexts.*


## Setup
```
# install COCO API
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# install SciPy
pip install scipy

# download MC-Bench evaltion code
git clone https://github.com/XuYunqiu/MC-Bench.git
```


## Data Preparation
### MC-Bench Dataset Download
* Download our dataset from [Google Drive](https://drive.google.com/drive/folders/1EWEiECUcJn4YD91z_n8FfHS1z5VKFi16?usp=drive_link) | [Baidu Pan](https://pan.baidu.com/s/1_e_rmUVgRy13ZVITlAOuXw?pwd=mc02)
* Unzip the `MC-Bench_images.zip` file
* The MC-Bench dataset directory should to have the following directory structure: 

```
MC-Bench/
├── mc-bench_v0.2_val.json/
└── mc-bench_images/
    ├── COCO/
    ├── D3/
    ├── winogavil_images/
    ├── BLINK_val_multiview/
    ├── Mantis-Eval_val/
    └── ...
```



### Annotation Format
The annotation format of MC-Bench is similar to COCO. The annotations are stored using JSON. The MC-Bench API can be used to access and manipulate annotations. The JSON file has the following format:
```
{
  "info"		:info,
  "images"		:[image],
  "annotations"		:[annotation],
  "descriptions"	:[description],
  "categories"		:categories,
}

description{
  "id"			:int,
  "images_id"		:[int],
  "text"		:str,
  "positive_sample"	:bool,
  "text_style"		:str,
}

image{
  "id"			:int,
  "text_id"		:int,
  "inter_img_id"	:int,
  "file_name"		:str,
  "height"		:int,
  "width"		:int,
}

annotation{
  "id"			:int,
  "image_id"		:int,
  "text_id"		:int,
  "catgory_id"		:int,
  "area"		:int,
  "bbox"		:[x,y,w,h],
  "iscrowd"		:0 or 1,
}
```
Note: box coordinates are floats measured from the top left image corner (and are 0-indexed). 


### Results Format
The results format is similar to the ground-truth annotation format
```
[{
  "image_id"		:int,
  "category_id"		:int,
  "bbox"		:[x,y,w,h],
  "score"		:float,
}]
```



## Evaluation in Command Line

To calculate the metrics used in MC-Bench, use
```
python eval_mc_bench.py \
  --gt_json_path ../MC-Bench/MC-Bench_coco_format.json.yaml \
  --dt_json_path /path/to/results_file \
  --eval_type 'all'
```
You can set `--eval_type` to `instance`, `image` or `all` for evaluation over instance-level, image-level and all metrics.



## TODO
- [ ] Refactor evalutaion code based on COCO API
- [ ] Support evaluation of more than 2 images
- [ ] MC-Bench v0.5



## Citation
If you find our data or code useful in your research, please use the following BibTeX entry.
```BibTeX
@article{xu2024mc,
  title={MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs},
  author={Xu, Yunqiu and Zhu, Linchao and Yang, Yi},
  journal={arXiv preprint arXiv:2410.12332},
  year={2024}
}
```


## Contact
If you have any questions, please drop me an email: imyunqiuxu@gmail.com
