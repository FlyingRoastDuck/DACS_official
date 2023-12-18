# demo for "XXX -> Market", you can decide target domain by changnig "-td" to specific dataset ("cuhk03", "market1501", "msmt17")

# FedPav (ResNet50)
CUDA_VISIBLE_DEVICES=1,0 python -W ignore fed_dg_cls.py -td market1501 --logs-dir ./logs/mar --data-dir ./data
# FedPav (ResNet50) + DACS
CUDA_VISIBLE_DEVICES=1,0 python -W ignore fed_dacs.py -td market1501 --logs-dir ./logs/mar --data-dir ./data

# FedPav (ViT)
CUDA_VISIBLE_DEVICES=1,0 python -W ignore fed_vit_dg.py -td market1501 --logs-dir ./logs/mar --data-dir ./data
# FedPav (ViT) + DACS
CUDA_VISIBLE_DEVICES=1,0 python -W ignore fed_vit_dacs.py -td market1501 --logs-dir ./logs/mar --data-dir ./data

# FedReID -- fed_reid.py