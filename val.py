from ultralytics import YOLO

# Load a pretrained YOLO11n model
print("::BEST::")
model = YOLO("ship12_experiment_lossbalance/1024res6/weights/last.pt")
metrics = model.val(data="../datasets/shipdatasets/DOTA_format_sup.yaml", device="0")


# experiences_temp/ship12_experiment_cls0/ship12-finetune-obb5/weights/best.pt 
# ../datasets/shipdatasets/DOTA_format_sup_cls0.yaml
# -> SHIP ONLY
# print("::cls0::")
# #model = YOLO("experiences_temp/ship12_experiment_cls0/ship12-finetune-obb5/weights/best.pt")
# model = YOLO("Whole_ship_pretrain/1024res/weights/last.pt")
# metrics = model.val(data="../datasets/shipdatasets/DOTA_format_sup_cls0.yaml", device="0")


# experiences_temp/ship12_experiment/ship12-finetune-obb2/weights/best.pt
# ../datasets/shipdatasets/DOTA_format_sup.yaml
# -> SHIP 12 class
#print("::cls12::")
#model = YOLO("experiences_temp/ship12_experiment/ship12-finetune-obb2/weights/best.pt")
#metrics = model.val(data="../datasets/shipdatasets/DOTA_format_sup.yaml", device="0")

