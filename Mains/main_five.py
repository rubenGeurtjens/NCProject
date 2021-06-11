""" 
Trains U-net two with color jitter
"""


from torchvision.transforms.transforms import RandomGrayscale
from Data_loader import ImageDataset, Data_loader
from U_net_two import UNet
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms

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

        dice_score = get_scores(val_loader, model, writer)
        if dice_score > best_score:
            best_score = dice_score
            #print("saving model...")
            save(OUTPUT_PATH+model.name, model)
            #print("saved succesfully")

        writer.write("-------------------------------\n")

def save(path, model):
    torch.save(model.state_dict(), path)

def save_images(loader, model, folder="masks2/", device="cuda"):
    folder = folder + model.name 
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device, dtype=torch.float)
        with torch.no_grad():
            outcome = torch.sigmoid(model(x))
            outcome = (outcome > 0.5).float()
        torchvision.utils.save_image(outcome, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{idx}.png")
    model.train()

def get_scores(loader, model, writer):
    n_pixels = 0
    dice_score = 0
    n_correct = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, dtype=torch.float)
            y = y.float().to(DEVICE).unsqueeze(1)
            outcome = torch.sigmoid(model(x))
            outcome = (outcome > 0.5).float()
            n_correct += (outcome == y).sum()
            n_pixels += torch.numel(outcome)
            dice_score += (2 * (outcome * y).sum()) / ((outcome + y).sum() + 1e-10
            )

    writer.write(f"Got {n_correct/n_pixels*100:.2f}% of the pixel classified correct\n")
    writer.write(f"Dice score: {dice_score/len(loader)} \n")
    model.train()
    return dice_score

if __name__ == "__main__":

    model = UNet().to(DEVICE)

    model.name += "_color_1000"

    #model.load_state_dict(torch.load('/home/ruben/Documents/school/Master 1/natural computing/NaturalComputing_assignments/results_project/results/models/u_net_two_color'))

    train_transform = transforms.Compose([
        transforms.ColorJitter(0.5,0.2,0.3,0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0],
                             std=[1.0,1.0,1.0])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0],
                             std=[1.0,1.0,1.0])
    ])


    writer = open(f"{OUTPUT_PATH+model.name}.txt", 'a+')
    writer.write(f"name: {model.name}\n\n")

    
    dataset = Data_loader(TRAIN_ANNOTATIONS_SMALL_PATH, TRAIN_IMAGES_DIRECTORY, 0.2, BATCH_SIZE, train_transform, val_transform, shuffle=True, seed=42, p=0.3 )
    train_loader, val_loader, data_set = dataset.get_loaders()
    train_idx = dataset.train_indices

    data_set.train_idx = train_idx

    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train(train_loader, val_loader, model, optimizer, loss, writer)
    writer.close()
    save_images(val_loader, model)