import cv2
import time
import math
import torch
import sys
from skimage import io
from torchvision import transforms, utils
import pygame
import random
import torchvision
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from queue import Queue
from threading import Thread
from PIL import Image
import matplotlib.pyplot as plt 

last_move = 0

def show_boxes(image, boxes, transpose=True):
    """Show image with landmarks"""
    plt.imshow(np.transpose(image, (1, 2, 0)) if transpose else image)
    ax = plt.gca()
    for box in boxes:
        h = box[3] - box[1]
        w = box[2] - box[0]
        rect = patches.Rectangle(
            (box[0], box[1]), w, h, linewidth=1, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)
    plt.pause(0.001)  # pause a bit so that plots are updated
 
 
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
 
num_classes = 2  # 1 class (hand) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
import torchvision.transforms as T
 
 
def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
 
 
checkpoint = torch.load("model_file", map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])
 
model.eval()

def predict():
    global last_move
    frame = image = io.imread("capture.jpg")
    prep = get_transform(False)
    trans = prep(frame)
    model.eval()
    trans_in = torch.stack([trans])
    
    t0 = time.time()
    output = model(trans_in)
    t1 = time.time()
    print(output)
    print(t1-t0)
    if len(output) == 0 or len(output[0]['boxes']) < 2:
        return
    box0 = output[0]['boxes'][0].cpu().data.numpy()
    box1 = output[0]['boxes'][1].cpu().data.numpy()
    box0_center = ((box0[0] + box0[2])/2, (box0[1] + box0[3])/2)
    box1_center = ((box1[0] + box1[2])/2, (box1[1] + box1[3])/2)
    # ensure box0_center is on the left
    if box0_center[0] > box1_center[0]:
        box0_center, box1_center = box1_center, box0_center
    opposite = box1_center[1] - box0_center[1]
    adjacent = box1_center[0] - box0_center[0]
    ang = math.atan(opposite/adjacent) * 180 / 3.14
    print('Angle: ', ang)
    if ang < -22:
        last_move = 1
    elif ang > 22:
        last_move = 2
    else:
        last_move = 0

# https://gist.github.com/cbednarski/8450931
def webcam():
    global last_move
    cap = cv2.VideoCapture(0)    
    i = 0
    while(True):
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        #frame = cv2.resize(frame, (480,270))
        #cv2.imshow('frame', rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            out = cv2.imwrite('capture.jpg', frame)
            predict()
            break
        if i % 2 == 0:
            out = cv2.imwrite('capture.jpg', frame)
            predict()
        i += 1
        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()


pygame.init()
width = 800
height = 600
size = (width,height)
fps = 120

screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()

font = pygame.font.Font('fonts/cargo.ttf',40)
score = pygame.font.Font('fonts/cargo.ttf',30)

background = pygame.image.load("game_images/roadway.jpg")
backrect = background.get_rect()
#author: Crack.com(bart) -> http://opengameart.org/content/golgotha-textures-tunnelroadjpg

carimg = pygame.image.load("game_images/car.png")
#author: shekh_tuhin -> http://opengameart.org/content/red-car-top-down
car_width = 49

truckimg = pygame.transform.scale(pygame.image.load("game_images/pickup.png"),(70,145))
#author: TRBRY -> http://opengameart.org/content/car-pickup

tires = pygame.mixer.Sound("sounds/tires_skid.ogg")
tires.set_volume(1)
#author: Mike Koenig (Soundbible) -> http://opengameart.org/content/car-tire-skid-squealing

crash = pygame.mixer.Sound("sounds/crash.ogg")
crash.set_volume(2)
#author: qubodup -> http://opengameart.org/content/crash-collision

countdown1 = pygame.mixer.Sound("sounds/countdown1.ogg")
countdown1.set_volume(1)
countdown1.play()

time.sleep(1)

countdown1 = pygame.mixer.Sound("sounds/countdown1.ogg")
countdown1.set_volume(1)
countdown1.play()

time.sleep(1)

countdown2 = pygame.mixer.Sound("sounds/countdown2.ogg")
countdown2.set_volume(1)
countdown2.play()
#author: Destructavator -> http://opengameart.org/content/countdown

time.sleep(1)

soundtrack = pygame.mixer.Sound("sounds/soundtrack.ogg")
soundtrack.set_volume(0.5)
soundtrack.play(-1)
#author: Dan Knoflicek -> http://opengameart.org/content/steppin-up

def avoided(count):
    scoreFont = score.render("Score: %d" % count, True, (0,0,0))
    screen.blit(scoreFont, (50,570))

def truck(truck_x,truck_y):
    screen.blit(truckimg,(truck_x,truck_y))

def car(x,y):
    screen.blit(carimg,(x,y))

def message2(x):
    messageFont2 = font.render("You hit a truck!", True, (0,0,0))
    rect = messageFont2.get_rect()
    rect.center = ((width//2),(height//2))
    screen.blit(messageFont2, rect)	
    pygame.display.update()    
    time.sleep(3)	
    playing()	
	
def message(x):
    messageFont = font.render("You went off the road!", True, (0,0,0))
    rect = messageFont.get_rect()
    rect.center = ((width//2),(height//2))
    screen.blit(messageFont, rect)    
    pygame.display.update()    
    time.sleep(3)    
    playing()	
	
def crashed2():
    message2("You hit a truck!")

def crashed():
    message("You went off the road!")
	
def playing():
    x = 351
    y = 480 	
    xChange = 0
    truck_x = random.randrange(50,770)
    truck_y = -500
    truck_speed = 2
    truck_height = 145
    truck_width = 70
    score = 0
    while True:
        clock.tick(fps)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    xChange = -6
                if event.key == pygame.K_RIGHT:
                    xChange = 6
                    
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                    xChange = 0
        
        if last_move == 0:
            xChange = 0
        elif last_move == 1:
            xChange = 0.3
        else:
            xChange = -0.3
        x += xChange

        screen.blit(background, backrect)

        truck(truck_x,truck_y)
        truck_y += truck_speed

        car(x,y)

        avoided(score)

        #crash detection if the car goes off the road
        if False and (x > (width - 87) or x < 35):
            tires.play()
            crash.play()
            crashed()

        #starting the truck along random coordinates
        if truck_y > height:
            truck_y =- 145
            truck_x = random.randrange(50,770)
            
            score += 1 #increase the score +1 for every truck is avoided
            truck_speed += 0.2 #increase the speed by 0.2 for every truck passed

        #collision detection for hitting the truck
        if False and y < truck_y + 145:
            
            if x > truck_x and x < truck_x + truck_width or x + car_width > truck_x and x + car_width < truck_x + truck_width:
                crash.play()
                crashed2()

        pygame.display.flip()

thread_cam = Thread(target=webcam)
thread_cam.start()
playing()
