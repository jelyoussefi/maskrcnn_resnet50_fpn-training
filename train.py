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
from torchvision import transforms as T

from tqdm import tqdm
import matplotlib.pyplot as plt

from model import Model
import warnings

warnings.filterwarnings("ignore")




#----------------------------------------------------------------------------------------------
# Data Set 
#----------------------------------------------------------------------------------------------
class CocoDataset(Dataset):
	def __init__(self, dataset_path, mode = 'train', augmentation=None):
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
		targets = {}
		targets["boxes"] =  boxes
		targets["labels"] = labels
		targets["masks"] = masks

		return image, targets

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
#  Training
#----------------------------------------------------------------------------------------------

def train(loader, model, optimizer, device):
	train_epoch_loss = 0
	loop = tqdm(loader)

	for batch_idx, (images, targets) in enumerate(loop):
		images = list(image.to(device) for image in images)
		targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
		if len(targets[0]['labels']) > 0:
			loss = model(images , targets)
			losses = sum([l for l in loss.values()])
			train_epoch_loss += losses.cpu().detach().numpy()
			optimizer.zero_grad()
			losses.backward()
			optimizer.step()
	return train_epoch_loss

#----------------------------------------------------------------------------------------------
#  Validation
#----------------------------------------------------------------------------------------------
    
def validate(loader, model, optimizer, device):
	val_epoch_loss = 0
	loop = tqdm(loader)
	with torch.no_grad():
		for batch_idx, (images, targets) in enumerate(loop):
			images = list(image.to(device) for image in images)
			targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
			if len(targets[0]['labels']) > 0:
				loss = model(images , targets)
				losses = sum([l for l in loss.values()])
				val_epoch_loss += losses.cpu().detach().numpy()
	return val_epoch_loss;

#----------------------------------------------------------------------------------------------
#  Main loop
#----------------------------------------------------------------------------------------------

def run(dataset_path, output_dir):

	device = "cuda" if torch.cuda.is_available() else "cpu"
	
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
	model = Model(num_classes)().to(device)
	optimizer = torch.optim.SGD(params=model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

	model.train()
	num_epocs = 50

	all_train_losses = []
	all_val_losses = []
	for epoch in range(30):
		train_epoch_loss = 0
		val_epoch_loss = 0
		model.train()
		train_epoch_loss = train(train_loader, model, optimizer, device)   
		all_train_losses.append(train_epoch_loss)
		valid_epoch_loss = valid(valid_loader, model, optimizer, device)
		all_val_losses.append(val_epoch_loss)
		print(epoch , "  " , train_epoch_loss , "  " , val_epoch_loss)

	save(model,output_dir)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default=None, help='dataset path')
	parser.add_argument('--output_dir', default="./model", help='outpur dir path')
	args = parser.parse_args()
	run(args.dataset, args.output_dir)
