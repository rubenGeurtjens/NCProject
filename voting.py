from sys import path_importer_cache
from main_four import OUTPUT_PATH
from PIL import Image
import numpy as np 
from matplotlib import pyplot as plt
import cv2

u_net_one_no_aug = "/home/ruben/Documents/school/Master 1/natural computing/NaturalComputing_assignments/results_project/saved_images/8366 images/u_net_one_no_aug"
u_net_one_gray = "/home/ruben/Documents/school/Master 1/natural computing/NaturalComputing_assignments/results_project/saved_images/8366 images/u_net_one_gray"
u_net_one_color = "/home/ruben/Documents/school/Master 1/natural computing/NaturalComputing_assignments/results_project/saved_images/8366 images/u_net_one_color"
u_net_two_gray = "/home/ruben/Documents/school/Master 1/natural computing/NaturalComputing_assignments/results_project/saved_images/8366 images/u_net_two_gray"
u_net_two_color = "/home/ruben/Documents/school/Master 1/natural computing/NaturalComputing_assignments/results_project/saved_images/8366 images/u_net_two_color"
u_net_three_gray = "/home/ruben/Documents/school/Master 1/natural computing/NaturalComputing_assignments/results_project/saved_images/8366 images/u_net_three_gray"
u_net_three_color = "/home/ruben/Documents/school/Master 1/natural computing/NaturalComputing_assignments/results_project/saved_images/8366 images/u_net_three_color"
u_net_four_gray = "/home/ruben/Documents/school/Master 1/natural computing/NaturalComputing_assignments/results_project/saved_images/8366 images/u_net_four_gray"
u_net_four_color = "/home/ruben/Documents/school/Master 1/natural computing/NaturalComputing_assignments/results_project/saved_images/8366 images/u_net_four_color"
masks = "/home/ruben/Documents/school/Master 1/natural computing/NaturalComputing_assignments/results_project/saved_images/8366 images/masks"

NR_PHOTOS = 1673

OUTCOME_PATH = "/home/ruben/Documents/school/Master 1/natural computing/NaturalComputing_assignments/project/ensemble_pics/"


accuracies = []
dices = []


dice_weights = [0.6724240183830261  , 0.7083433270454407 , 0.6186125874519348, 0.2812654674053192, 0.3351667821407318, 0.5855615139007568,  0.6924539804458618, 0.6259531378746033, 0.6556903123855591 ]

acc_weights = [88.40, 88.61, 87.45, 80.13, 87.79, 87.28, 88.69, 83.75, 85.05]

normalized_acc_weights = [a/sum(acc_weights) for a in acc_weights]
normalized_dice_weights = [d/sum(dice_weights) for d in dice_weights]


correlations = np.zeros((9,9))
amount = 0 

for idx in range(NR_PHOTOS):
    pic1_no_aug = cv2.imread(u_net_one_no_aug + "/pred_" + str(idx) + ".png")[:,:,0] / 255
    pic1_gray = cv2.imread(u_net_one_gray + "/pred_" + str(idx) + ".png")[:,:,0] / 255
    pic1_color = cv2.imread(u_net_one_color + "/pred_" + str(idx) + ".png")[:,:,0] / 255
    pic2_gray = cv2.imread(u_net_two_gray + "/pred_" + str(idx) + ".png")[:,:,0] / 255
    pic2_color = cv2.imread(u_net_two_color + "/pred_" + str(idx) + ".png")[:,:,0] / 255
    pic3_gray = cv2.imread(u_net_three_gray + "/pred_" + str(idx) + ".png")[:,:,0] / 255
    pic3_color = cv2.imread(u_net_three_color + "/pred_" + str(idx) + ".png")[:,:,0] / 255
    pic4_gray = cv2.imread(u_net_four_gray + "/pred_" + str(idx) + ".png")[:,:,0] / 255
    pic4_color = cv2.imread(u_net_four_color + "/pred_" + str(idx) + ".png")[:,:,0] / 255


    mask = cv2.imread(masks + "/" + str(idx) + ".png")[:,:,0] / 255

    #mayority voting

    pred = pic1_no_aug + pic1_gray + pic1_color + pic2_gray + pic2_color + pic3_gray + pic3_color + pic4_gray + pic4_color
    pred = np.floor(pred / 5)

    #weighted mayority voting on accuracy

    colors = [pic1_no_aug , pic1_gray , pic1_color , pic2_gray , pic2_color , pic3_gray , pic3_color , pic4_gray , pic4_color]
    pred = np.zeros_like(pic1_color)
    for w,i in zip(normalized_acc_weights, colors):
        pred += w*i
    pred[pred >= 0.5] = 1
    pred[pred < 0.5 ] = 0

    #weighted mayority voting on dice

    colors = [pic1_no_aug , pic1_gray , pic1_color , pic2_gray , pic2_color , pic3_gray , pic3_color , pic4_gray , pic4_color]
    pred = np.zeros_like(pic1_color)
    for w,i in zip(normalized_dice_weights, colors):
        pred += w*i
    pred[pred >= 0.5] = 1
    pred[pred < 0.5 ] = 0


    ##calculating accuracy

    w,h = mask.shape
    accuracies.append((pred == mask).sum() / (w*h))

    ##calculating dice

    dice = 2*(pred * mask).sum() / ((pred + mask).sum() + 1e-8)
    dices.append(dice)

    #used to save images

    loc = OUTCOME_PATH+"pred_" + str(idx)+ ".jpg"
    print(loc)
    cv2.imwrite(loc, pred)

    ##used to calculate correlations

    do_cor = True
    for pic in colors:
        if pic.flatten().sum() == 0 or pic.flatten().sum() == 300*300:
            do_cor = False
        
    if do_cor:
        amount += 1
        cor = np.corrcoef([pic1_no_aug.flatten() , pic1_gray.flatten() , pic1_color.flatten() , pic2_gray.flatten() , pic2_color.flatten() , pic3_gray.flatten() , pic3_color.flatten() , pic4_gray.flatten() , pic4_color.flatten()])
        correlations += cor

    #get agreement between all models

    pred = pic1_no_aug + pic1_gray + pic1_color + pic2_gray + pic2_color + pic3_gray + pic3_color + pic4_gray + pic4_color
    pred[pred == 9] = 1
    pred[pred == 0] = 1 
    pred[pred != 1] = 0
    amount += pred.flatten().sum()

print(np.mean(accuracies))
print(np.mean(dices))
output = correlations/amount
print(np.round(output, 2))
print(amount/(300*300*NR_PHOTOS))
