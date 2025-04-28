# yolov12_additional_func

# Training Analysis about obb detection
'''
2025_04_28
1-Class Validation
- ultralystic/cfg/defaults
- val_cls_agnostic: True # True for class agnostic mAP! which means GT's and Pred's class is -> 0 ( every object's cls = 0 )

Topk Validation
- ultralystic/cfg/defaults
- val_topk: 3 # None or 2, 3 will be top2 or top3 confidence based mAP
'''
