""" 
Used to get the output from all unets on the dataset in the same order. Doesn't work otherwise because pytorch is not deterministic. 
"""

from torchvision.transforms.transforms import Grayscale, RandomGrayscale
from Data_loader import ImageDataset, Data_loader

from U_net_one import UNet as UNet1
from U_net_two import UNet as UNet2
from U_net_three import UNet as UNet3
from U_net_four import UNet as UNet4 

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms
import numpy
import random 

TRAIN_IMAGES_DIRECTORY = "small_data" # data/train/images"
TRAIN_ANNOTATIONS_PATH = "data/train/annotation.json" #roughly 280000 images
TRAIN_ANNOTATIONS_SMALL_PATH = "data/train/annotation-small.json" #8366 images

BATCH_SIZE = 1
EPOCHS = 3
LEARNING_RATE = 1e-4 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_PATH = "results/"


torch.manual_seed(200)
torch.backends.cudnn.deterministic = True # Not necessary in this example
torch.cuda.manual_seed_all(200)
random.seed(200) 
np.random.seed(42)
torch.backends.cudnn.benchmark = False

def train(train_loader, val_loader, model, optimizer, criterion, writer):
    best_score = 0
    for epoch in range(EPOCHS):
        avg_loss = []
        i = 0
        for batch_idx, (data, mask) in enumerate(train_loader):
            data = data.to(DEVICE, dtype=torch.float)
            mask =  mask.float().unsqueeze(0).to(DEVICE)

            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, mask)
            loss.backward()

            avg_loss.append(loss.item())
            optimizer.step()
            # if i % 25  == 0:
            #    print(f"{i/len(train_loader)*100:.2f}%")
            #i+=1

        writer.write(f"epoch: {epoch}\n")
        writer.write(f"train loss: {np.mean(avg_loss)}\n")

        dice_score = eval_model(val_loader, model, writer)
        if dice_score > best_score:
            best_score = dice_score
            #print("saving model...")
            save(OUTPUT_PATH+model.name, model)
            #print("saved succesfully")

        writer.write("-------------------------------\n")

def save(path, model):
    torch.save(model.state_dict(), path)

def save_img(loader, models, folder="saved_images/", device="cuda"):


    for idx, (x, y) in enumerate(loader):
        all_masks = True

        for model in models:
            folder = "saved_images/" + model.name 
            #model.eval()

            x = x.to(device=device, dtype=torch.float)
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
            torchvision.utils.save_image(
                preds, f"{folder}/pred_{idx}.png"
            )
            if all_masks:
                torchvision.utils.save_image(y.unsqueeze(1), f"saved_images/masks/{idx}.png")

            all_masks = False
    #model.train()

def eval_model(loader, model, writer):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, dtype=torch.float)
            y = y.float().to(DEVICE).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    writer.write(f"Got {num_correct/num_pixels*100:.2f}% of the pixel classified correct\n")
    writer.write(f"Dice score: {dice_score/len(loader)} \n")
    #print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    #print(f"Dice score: {dice_score/len(loader)}")
    model.train()
    return dice_score

if __name__ == "__main__":
    
    model1 = UNet1().to(DEVICE)
    model1.name += "_no_aug_1000"
    model1.load_state_dict(torch.load('/home/ruben/Documents/school/Master 1/natural computing/NaturalComputing_assignments/results_project/results/1000 images/models/u_net_one_no_aug_1000'))
    model1.eval()

    model2 = UNet1().to(DEVICE)
    model2.name += "_gray_1000"
    model2.load_state_dict(torch.load('/home/ruben/Documents/school/Master 1/natural computing/NaturalComputing_assignments/results_project/results/1000 images/models/u_net_one_gray_1000'))
    model2.eval()

    model3 = UNet1().to(DEVICE)
    model3.name += "_color_1000"
    model3.load_state_dict(torch.load('/home/ruben/Documents/school/Master 1/natural computing/NaturalComputing_assignments/results_project/results/1000 images/models/u_net_one_color_1000'))
    model3.eval()

    model4 = UNet2().to(DEVICE)
    model4.name += "_gray_1000"
    model4.load_state_dict(torch.load('/home/ruben/Documents/school/Master 1/natural computing/NaturalComputing_assignments/results_project/results/1000 images/models/u_net_two_gray_1000'))
    model4.eval()

    model5 = UNet2().to(DEVICE)
    model5.name += "_color_1000"
    model5.load_state_dict(torch.load('/home/ruben/Documents/school/Master 1/natural computing/NaturalComputing_assignments/results_project/results/1000 images/models/u_net_two_color_1000'))
    model5.eval()

    model6 = UNet3().to(DEVICE)
    model6.name += "_gray_1000"
    model6.load_state_dict(torch.load('/home/ruben/Documents/school/Master 1/natural computing/NaturalComputing_assignments/results_project/results/1000 images/models/u_net_three_gray_1000'))
    model6.eval()

    model7 = UNet3().to(DEVICE)
    model7.name += "_color_1000"
    model7.load_state_dict(torch.load('/home/ruben/Documents/school/Master 1/natural computing/NaturalComputing_assignments/results_project/results/1000 images/models/u_net_three_color_1000'))
    model7.eval()

    model8 = UNet4().to(DEVICE)
    model8.name += "_gray_1000"
    model8.load_state_dict(torch.load('/home/ruben/Documents/school/Master 1/natural computing/NaturalComputing_assignments/results_project/results/1000 images/models/u_net_four_gray_1000'))
    model8.eval()

    model9 = UNet4().to(DEVICE)
    model9.name += "_color_1000"
    model9.load_state_dict(torch.load('/home/ruben/Documents/school/Master 1/natural computing/NaturalComputing_assignments/results_project/results/1000 images/models/u_net_four_gaus_1000'))
    model9.eval()

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0],
                             std=[1.0,1.0,1.0])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0],
                             std=[1.0,1.0,1.0])
    ])


    writer = open(f"{OUTPUT_PATH+model1.name}.txt", 'a+')
    writer.write(f"name: {model1.name}\n\n")

    
    dataset = Data_loader(TRAIN_ANNOTATIONS_SMALL_PATH, TRAIN_IMAGES_DIRECTORY, 0.2, BATCH_SIZE, train_transform, val_transform, shuffle=True, seed=42, p=1)
    train_loader, val_loader, data_set = dataset.get_loaders()
    train_idx = dataset.train_indices

    data_set.train_idx = train_idx

    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model1.parameters(), lr=LEARNING_RATE)

    #train(train_loader, val_loader, model, optimizer, loss, writer)
    writer.close()

    models = [model1,model2,model3,model4,model5,model6,model7,model8,model9]

    save_img(val_loader, models)
