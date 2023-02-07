import os
import copy
import random
import cv2
import torch
import numpy as np
import pandas as pd
from torch import nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from ipywidgets import interact

random_seed = 2022

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def list_image_files(data_dir, sub_dir):
    image_format = ["jpeg", "jpg", "png"] # 이미지 확장자
    
    image_files = []
    images_dir = os.path.join(data_dir, sub_dir)
    for file_path in os.listdir(images_dir):
        if file_path.split(".")[-1] in image_format: # 확장자가 올바르면 이미지 가져오기
            image_files.append(os.path.join(sub_dir, file_path))
    return image_files

data_dir = "./dataset"

custom_list = list_image_files(data_dir, "custom") # custom image

model = models.vgg19(pretrained=True) # vgg19 pretrained 모델 사용

# face dataset에 맞게 모델 아키텍처 수정
# output 사이즈를 지정하여 연산을 수행할 수 있음
class_list = ['real', 'fake']
model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
model.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(512, 256), # fully-connected
    nn.ReLU(), # activation function
    nn.Dropout(0.1), # 정규화
    nn.Linear(256, len(class_list)), # real, fake 두 개 클래스
    nn.Sigmoid() # 시그모이 함수로 확률 출력
)

# pretrained vgg19 모델에 avgpool과 classifier를 추가하여 새로운 model 생성
def build_vgg19_based_model(device_name='cuda'): 
    device = torch.device(device_name)
    model = models.vgg19(pretrained=True)
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, len(class_list)),
        nn.Softmax(dim=1)
    )
    return model.to(device)

def get_RGB_image(data_dir, file_name): 
    image_file = os.path.join(data_dir, file_name) # 상위 경로 + 하위 경로
    image = cv2.imread(image_file) # opencv로 이미지를 읽어옴(bgr 형식)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # bgr 형식을 rgb 형식으로 변환
    return image

def preprocess_image(image):
    ori_H, ori_W = image.shape[:2]
    
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
        ])
    
    tensor_image = transformer(image)
    tensor_image = tensor_image.unsqueeze(0)
    return tensor_image

ckpt = torch.load("./model/model_20.pth")

model = build_vgg19_based_model(device_name="cpu")
model.load_state_dict(ckpt)
model.eval()

class_correct = list(0. for i in range(1))
class_total = list(0. for i in range(1))

def test_model(image, model):
    tensor_image = preprocess_image(image)

    with torch.no_grad():
        prediction = model(tensor_image)

    _, pred_label = torch.max(prediction.detach(), dim=1)

    pred_label = pred_label.squeeze(0)

    # 이미지가 real일 확률
    prob_list = prediction.tolist()
    prob_reduce_list = np.array(prob_list).flatten().tolist()    
    #print(prob_reduce_list[0])

    return pred_label.item(), prob_reduce_list[0]

data_data = "./dataset/custom"
class_list = ["real", "fake"]

test_customs_list = list_image_files(data_dir, "custom")
min_num_files = len(test_customs_list)

@interact(index=(0, min_num_files-1))
def show_result(index=0):
    custom_image = get_RGB_image(data_dir, test_customs_list[index])

    prediction_1, prob1 = test_model(custom_image, model)

    prob1 = format(prob1, '.3f')

    plt.figure(figsize=(12,8))
    plt.subplot(131)
    plt.title(f"Pred:{class_list[prediction_1]} | GT:real | Prob:{prob1}")
    plt.imshow(custom_image)
    plt.tight_layout()