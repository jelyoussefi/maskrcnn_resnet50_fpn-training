import os,sys,time,argparse
import numpy as np
import math
import cv2
import torch
import torchvision
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
import matplotlib.pyplot as plt

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.models.detection.mask_rcnn
import warnings

warnings.filterwarnings("ignore")




#----------------------------------------------------------------------------------------------
# Data Set 
#----------------------------------------------------------------------------------------------
class CocoDataset(Dataset):
	def __init__(self, dataset_path, mode = 'train', augmentation=None):
		if mode == 'train':
			self.dataset_path = os.path.join(dataset_path, mode)
			ann_path = os.path.join(self.dataset_path, '_annotations.coco.json')
			self.coco = COCO(ann_path)
			self.cat_ids = self.coco.getCatIds()
			self.augmentation = augmentation

	def __len__(self):
		return len(self.coco.imgs)
	  
	def get_masks(self, index):
		ann_ids = self.coco.getAnnIds([index])
		anns = self.coco.loadAnns(ann_ids)
		masks=[]
      
		for ann in anns:
			mask = self.coco.annToMask(ann)
			masks.append(mask)

		return masks

	def get_boxes(self, masks):
		num_objs = len(masks)
		boxes = []

		for i in range(num_objs):
			x,y,w,h = cv2.boundingRect(masks[i])
			boxes.append([x, y, x+w, y+h])

		return np.array(boxes)

	def __getitem__(self, index):
		# Load image
		img_info = self.coco.loadImgs([index])[0]
		image = cv2.imread(os.path.join(self.dataset_path,
		                            img_info['file_name']))
		masks = self.get_masks(index)

		if self.augmentation:
			augmented = self.augmentation(image=image, masks=masks)
			image, masks = augmented['image'], augmented['masks']

		image = image.transpose(2,0,1) / 255.

		# Load masks
		masks = np.array(masks)
		boxes = self.get_boxes(masks)

		# Create target dict
		num_objs = len(masks)
		boxes = torch.as_tensor(boxes, dtype=torch.float32)
		labels = torch.ones((num_objs,), dtype=torch.int64)
		masks = torch.as_tensor(masks, dtype=torch.uint8)
		image = torch.as_tensor(image, dtype=torch.float32)
		data = {}
		data["boxes"] =  boxes
		data["labels"] = labels
		data["masks"] = masks

		return image, data

#----------------------------------------------------------------------------------------------
# Data Augmentation
#----------------------------------------------------------------------------------------------
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(
        contrast_limit=0.2, brightness_limit=0.3, p=0.5),
    A.OneOf([
        A.ImageCompression(p=0.8),
        A.RandomGamma(p=0.8),
        A.Blur(p=0.8),
        A.Equalize(mode='cv',p=0.8)
    ], p=1.0),
    A.OneOf([
        A.ImageCompression(p=0.8),
        A.RandomGamma(p=0.8),
        A.Blur(p=0.8),
        A.Equalize(mode='cv',p=0.8),
    ], p=1.0)
])


#----------------------------------------------------------------------------------------------
# Utils
#----------------------------------------------------------------------------------------------
def collate_fn(batch):
	images = list()
	targets = list()
	for b in batch:
		images.append(b[0])
		targets.append(b[1])
	images = torch.stack(images, dim=0)
	return images, targets

def save(model, output_dir):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	date_time = datetime.now().strftime("%d_%m_%Y_%H_%M")
	model_path=os.path.join(output_dir,"model_"+date_time+".pth")
	print("\nSaving ", model_path, " model ...\n")
	torch.save(model.state_dict(), model_path)
	model_path_symblink = os.path.join(output_dir, "best.pth")
	if os.path.islink(model_path_symblink):
		os.remove(model_path_symblink)
	os.symlink(os.path.basename(model_path), model_path_symblink)

#----------------------------------------------------------------------------------------------
# Model
#----------------------------------------------------------------------------------------------
def get_model_instance_segmentation(num_classes):
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


#----------------------------------------------------------------------------------------------
#  Training
#----------------------------------------------------------------------------------------------
def train_one_epoch(loader, model, optimizer, device):
	loop = tqdm(loader)

	for batch_idx, (images, targets) in enumerate(loop):
		images = list(image.to(device) for image in images)
		targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

		loss_dict = model(images, targets)
		losses = sum(loss for loss in loss_dict.values())

		optimizer.zero_grad()
		losses.backward()
		optimizer.step()

	print(f"Total loss: {losses.item()}")

#----------------------------------------------------------------------------------------------
#  Validation
#----------------------------------------------------------------------------------------------
best_vloss = np.inf
def validate(loader, model, optimizer, device, epoch):
    global best_vloss
    loop = tqdm(loader)
    running_vloss = 0
    for batch_idx, (images, targets) in enumerate(loop):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
          loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())
        running_vloss += losses
        
    avg_vloss = running_vloss / (batch_idx + 1)
    
    print(f"Avg Valid Loss: {avg_vloss}")
    if avg_vloss < best_vloss:
      best_vloss = avg_vloss
      if SAVE_MODEL:
            print("Model improved, saving...")
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            #save_checkpoint(checkpoint, filename=f"1152KaggleBest_second_{epoch}.pth.tar")
    print('\n')
    return avg_vloss


def train_one_epoch(loader, model, optimizer, device):
    loop = tqdm(loader)

    for batch_idx, (images, targets) in enumerate(loop):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        

def run(dataset_path, output_dir):

	device = "cuda" if torch.cuda.is_available() else "cpu"
	lr_rate = 1e-5
	weight_decay = 5e-4
	batch_size = 2
	image_size = [640,640]

	train_dataset = CocoDataset(dataset_path, mode='train', augmentation=None)
	train_loader = DataLoader(	dataset=train_dataset,
								batch_size=batch_size,
								shuffle=True,
								num_workers=2,
								pin_memory=True,
								collate_fn=collate_fn)

	valid_dataset = CocoDataset(dataset_path, mode='valid')
	valid_loader = DataLoader(	dataset=valid_dataset,
								batch_size=batch_size,
								shuffle=False,
								pin_memory=True,
								collate_fn=collate_fn)

	num_classes = len(train_dataset.cat_ids)
	model = get_model_instance_segmentation(num_classes)
	model.to(device)
	optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr_rate, weight_decay=weight_decay)

	model.train()
	num_epocs = 50
	for epoch in range(num_epocs):
		print(f"Epoch: {epoch}")
		train_one_epoch(train_loader, model, optimizer, device)
		vloss= validate(valid_loader, model, optimizer, device, epoch)

	save(model,output_dir)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default=None, help='dataset path')
	parser.add_argument('--output_dir', default="./model", help='outpur dir path')
	args = parser.parse_args()
	run(args.dataset, args.output_dir)
