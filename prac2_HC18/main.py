import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class HC18Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".png","_Annotation.png"))
        image = np.array(Image.open(img_path).resize((800,540)).convert("RGB"))
        image = np.transpose(image, (2, 0, 1))

        #print(image.shape)
        mask = np.array(Image.open(mask_path).resize((800,540)).convert("L"), dtype =np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return torch.Tensor(image), torch.FloatTensor(mask)


train_data = HC18Dataset("/kaggle/input/hc18-split/training_set","/kaggle/input/hc18-split/training_mask")
val_data = HC18Dataset("/kaggle/input/hc18-split/val_set","/kaggle/input/hc18-split/val_mask")


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,1,1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3,1,1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        #print(f'Input shape:     {x.shape}')
        x = self.conv(x)
        #print(f'Output shape:    {x.shape}')
        return x


class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Down part
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        #Up part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature,kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups),2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size = skip_connection.shape[2:], antialias=None)

            concat_skip = torch.cat((skip_connection, x), dim =1)
            x = self.ups[idx+1](concat_skip) 
        return self.final_conv(x)
    

# def test():
#     x = torch.randn((1, 1, 572, 572))
#     model = UNET(in_channels=1, out_channels=1)
#     preds = model(x)
#     #print(model)
#     #print(preds)
# test()
    
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMAGE_HEIGHT = 80  # 540 originally
IMAGE_WIDTH = 120  # 800 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "/kaggle/input/hc18-split/training_set"
TRAIN_MASK_DIR = "/kaggle/input/hc18-split/training_mask"
VAL_IMG_DIR = "/kaggle/input/hc18-split/val_set"
VAL_MASK_DIR = "/kaggle/input/hc18-split/val_mask"


model = UNET(in_channels=3, out_channels=1).to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )
val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )


def train_fn(loader, model, optimizer, loss_fn):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        predictions = model(data)
        loss = loss_fn(predictions, targets)
        #print(f'************{loss}')
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        loop.set_postfix(loss=loss.item())
    return loss.item()
def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    IOU_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            IOU_score += ((preds * y).sum()) / (
                ((preds + y)>0).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"IOU score: {IOU_score/len(loader)}")
    model.train()
    return IOU_score
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


loss_accumulate = []
IOU_list = []
IOU_threshold = 0
for epoch in range(NUM_EPOCHS):
    loss_accumulate.append(train_fn(train_loader, model, optimizer, loss_fn))
    IOU_ = check_accuracy(val_loader, model, device=DEVICE)
    checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
    if IOU_ > IOU_threshold:
        IOU_threshold = IOU_
        save_checkpoint(checkpoint)
    IOU_list.append(IOU_)
