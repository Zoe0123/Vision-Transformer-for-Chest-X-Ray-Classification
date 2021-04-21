import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from grad_cam import grad_cam

import torch
from coronahack import ResNetV2, StdConv2d, PreActBottleneck

def main():
    corona_bit_example()
    corona_vit_example()
    # boxer_example()
    # tiger_cat_example()
    # elephant_example()

def boxer_example():
    model = torchvision.models.resnet34(pretrained=True)
    model.eval()
    print(model.features)
    transform = transforms.Compose([
        transforms.Resize(240),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    heatmap_layer = model.layer4[2].conv2
    image = Image.open("./images/cat_dog.png")
    input_tensor = transform(image)
    boxer_label = 242
    image = grad_cam(model, input_tensor, heatmap_layer, boxer_label)
    plt.imshow(image)
    plt.savefig('./images/boxer_grad-cam')

def tiger_cat_example():
    model = torchvision.models.resnet34(pretrained=True)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(240),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    heatmap_layer = model.layer4[2].conv2
    image = Image.open("./images/cat_dog.png")
    input_tensor = transform(image)
    tiger_cat_label = 282
    image = grad_cam(model, input_tensor, heatmap_layer, tiger_cat_label)
    plt.imshow(image)
    plt.savefig('./images/tiger_cat_grad-cam')

def elephant_example():
    model = torchvision.models.resnet34(pretrained=True)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(240),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    heatmap_layer = model.layer4[2].conv2
    image = Image.open("./images/elephant.jpg")
    input_tensor = transform(image)
    elephant_label = 386
    image = grad_cam(model, input_tensor, heatmap_layer, elephant_label)
    plt.imshow(image)
    plt.savefig('./images/elephant_grad-cam')

def corona_bit_example():
    model = torch.load("../bitm_20.pth")
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(240),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    heatmap_layer = model.body.block4.unit03.conv3

    img_path_n = "./images/example_n.JPG"
    image_n = Image.open(img_path_n)
    input_tensor_n = transform(image_n)
    target_class_n = 0
    image_n_hm = grad_cam(model, input_tensor_n, heatmap_layer, target_class_n)
    plt.imshow(image_n_hm)
    plt.savefig('./images/bit_n_hm.png')

    img_path_v = "./images/example_v.JPG"
    image_v = Image.open(img_path_v)
    input_tensor_v = transform(image_v)
    target_class_v = 1
    image_v_hm = grad_cam(model, input_tensor_v, heatmap_layer, target_class_v)
    plt.imshow(image_v_hm)
    plt.savefig('./images/bit_v_hm.png')

def corona_vit_example():
    model = torch.load("../vit_10.pth", map_location=torch.device('cpu'))
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(240),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    heatmap_layer = model.patch_embed.backbone.stages[2].blocks[8].conv3

    img_path_n = "./images/example_n.JPG"
    image_n = Image.open(img_path_n)
    input_tensor_n = transform(image_n)
    target_class_n = 0
    image_n_hm = grad_cam(model, input_tensor_n, heatmap_layer, target_class_n)
    plt.imshow(image_n_hm)
    plt.savefig('./images/vit_n_hm.png')

    img_path_v = "./images/example_v.JPG"
    image_v = Image.open(img_path_v)
    input_tensor_v = transform(image_v)
    target_class_v = 1
    image_v_hm = grad_cam(model, input_tensor_v, heatmap_layer, target_class_v)
    plt.imshow(image_v_hm)
    plt.savefig('./images/vit_v_hm.png')

if __name__== "__main__":
    main()
    print("done")