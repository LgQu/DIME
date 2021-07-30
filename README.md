# Dynamic Modality Interaction Modeling for Image-Text Retrieval (DIME)

PyTorch code of the paper "Dynamic Modality Interaction Modeling for Image-Text Retrieval". It is built on top of [VSRN](https://github.com/KunpengLi1994/VSRN) and [CAMERA](https://acmmmcamera.wixsite.com/camera). 

## Introduction
Image-text retrieval is a fundamental and crucial branch in information retrieval. Although much progress has been made in bridging vision and language, it remains challenging because of the difficult *intra-modal reasoning* and *cross-modal alignment*. Existing modality interaction methods have achieved impressive results on public datasets, however, they heavily rely on expert experience and empirical feedback towards the design of interaction patterns, therefore, lacking flexibility.
To address these issues, we develop a novel modality interaction modeling network based upon the routing mechanism, which is the first unified and dynamic multimodal interaction framework towards image-text retrieval. In particular, we first design four types of cells as basic units to explore different levels of modality interactions, and then connect them in a dense strategy to construct a routing space. To endow the model with the capability of path decision, we integrate a dynamic router in each cell for pattern exploration. As the routers are conditioned on inputs, our model can dynamically learn different activated paths for different data. Extensive experiments on two benchmark datasets, *i.e*., Flickr30K and MS-COCO, verify the superiority of our model compared with several state-of-the-art baselines. 

![model](/fig/model.png)

## Requirements 
We recommended the following dependencies.

* Python 2.7 
* [PyTorch](http://pytorch.org/) (1.0.1)
* [NumPy](http://www.numpy.org/) (>=1.16.5)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)
* [pycocotools](https://github.com/cocodataset/cocoapi)
* [torchvision]()
* [matplotlib]()

## Download data

Download the dataset files and pre-trained models. We use splits produced by [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/deepimagesent/). The raw images can be downloaded from from their original sources [here](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html), [here](http://shannon.cs.illinois.edu/DenotationGraph/) and [here](http://mscoco.org/).

We follow [bottom-up attention model](https://github.com/peteanderson80/bottom-up-attention) and [SCAN](https://github.com/kuanghuei/SCAN) to obtain image features for fair comparison. More details about data pre-processing (optional) can be found [here](https://github.com/kuanghuei/SCAN/blob/master/README.md#data-pre-processing-optional). All the precomputed image features data needed for reproducing the experiments in the paper, can be downloaded from [SCAN](https://github.com/kuanghuei/SCAN) by using:

```bash
wget https://scanproject.blob.core.windows.net/scan-data/data.zip
```

You can also get the data from google drive: https://drive.google.com/drive/u/1/folders/1os1Kr7HeTbh8FajBNegW8rjJf6GIhFqC. 

We refer to the path of extracted files as `$DATA_PATH`. 

## BERT model

We use the BERT code from [BERT-pytorch](https://github.com/huggingface/transformers). Please following [here](https://github.com/huggingface/pytorch-transformers/blob/4fc9f9ef54e2ab250042c55b55a2e3c097858cb7/docs/source/converting_tensorflow_models.rst) to convert the Google BERT model to a PyTorch save file `$BERT_PATH`.

## Train new models
Run `train.py`:

For DIME (i-t) on MSCOCO:

```bash
python train.py --data_path $DATA_PATH --bert_path $BERT_PATH --data_name coco_precomp --logger_name runs/coco_DIME --direction i2t --extra_stc 1 --lambda_softmax=4
```

For DIME (t-i) on MSCOCO:

```bash
python train.py --data_path $DATA_PATH --bert_path $BERT_PATH --data_name coco_precomp --logger_name runs/coco_DIME --direction t2i --extra_img 1 --lambda_softmax=9
```

For DIME (i-t) on Flickr30K:

```bash
python train.py --data_path $DATA_PATH --bert_path $BERT_PATH --data_name f30k_precomp --logger_name runs/filker_DIME --direction i2t --extra_stc 1 --lambda_softmax=4
```

For DIME (i-t) on Flickr30K:

```bash
python train.py --data_path $DATA_PATH --bert_path $BERT_PATH --data_name f30k_precomp --logger_name runs/filker_DIME --direction t2i --extra_img 1 --lambda_softmax=9
```

## Evaluate trained models

Modify the model_path and data_path in the evaluation_models.py file. Then Run `evaluate_models.py`:

```bash
python evaluate_models.py
```

To do cross-validation on MSCOCO 1K test set, pass `fold5=True`. Pass `fold5=False` for evaluation on MSCOCO 5K test set. Pretrained models can be downloaded from https://drive.google.com/file/d/1vurSb_Ssr2wlKaJCA5Vu5w6I3ee17RFM/view?usp=sharing.

## Reference

```
@inproceedings{qu2021dynamic,
  title={Dynamic Modality Interaction Modeling for Image-Text Retrieval},
  author={Qu, Leigang and Liu, Meng and Wu, Jianlong and Gao, Zan and Nie, Liqiang},
  booktitle={Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={1104--1113},
  year={2021}
}
```

