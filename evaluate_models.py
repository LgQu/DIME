import torch
import evaluation_models


# flickr
evaluation_models.evalrank_single("pretrain_model/flickr/Flickr30K_DIME_it.pth.tar", data_path='/to/data/path/', split="test", fold5=False)
# evaluation_models.evalrank_ensemble("pretrain_model/flickr/Flickr30K_DIME_it.pth.tar", "pretrain_model/flickr/Flickr30K_DIME_ti.pth.tar", \
#                     data_path='/to/data/path/', split="test", fold5=False)

# coco
# evaluation_models.evalrank_single("pretrain_model/coco/MSCOCO_DIME_it.pth.tar", data_path='/to/data/path/', split="testall", fold5=True)
# evaluation_models.evalrank_ensemble("pretrain_model/coco/MSCOCO_DIME_it.pth.tar", "pretrain_model/coco/MSCOCO_DIME_ti.pth.tar", \
#                     data_path='/to/data/path/', split="testall", fold5=True)



