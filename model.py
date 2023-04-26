from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.models.detection.mask_rcnn

class Model:
	def __init__(self, num_classes):
		# Load an instance segmentation model pre-trained on COCO
		self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

		# Get number of input features for the classifier
		in_features = self.model.roi_heads.box_predictor.cls_score.in_features
		# replace the pre-trained head with a new one
		self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

		# Get the number of input features for the mask classifier
		in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
		hidden_layer = 256
		# Replace the mask predictor with a new one
		self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
		                                                   		hidden_layer,
		                                                   		num_classes)

							 
	def __call__(self):
		return self.model


