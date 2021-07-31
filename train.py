import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# load a model pre-trained on COCO
# 从coco训练集加载其预训练模型
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# replace the classifier with a new one, that has
# num_classes which is user-defined
# 两个类别
num_classes = 2  # 1 class (person) + background
# get number of input features for the classifier
# 得到输入特征
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
# 替换为自己的类别数和特征
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
