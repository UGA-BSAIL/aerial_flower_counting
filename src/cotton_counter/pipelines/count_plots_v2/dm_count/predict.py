import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from models import vgg19, yolov8
from torchvision import transforms

model_path = "./best_model_6.pth"
device = torch.device("cuda")  # device can be "cpu" or "gpu"

# model = vgg19()
model = yolov8()
model.to(device)
model.load_state_dict(torch.load(model_path, device))
model.eval()


def predict(inp):
    # inp = transforms.ToTensor()(inp).unsqueeze(0)
    process = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    inp = process(inp).unsqueeze(0)
    inp = inp.to(device)
    with torch.set_grad_enabled(False):
        outputs0, _ = model(inp)
    count = torch.sum(outputs0).item()

    vis_img0 = outputs0[0, 0].cpu().numpy()
    # normalize density map values from 0 to 1, then map it to 0-255.
    vis_img0 = (vis_img0 - vis_img0.min()) / (
        vis_img0.max() - vis_img0.min() + 1e-5
    )
    vis_img0 = (vis_img0 * 255).astype(np.uint8)
    vis_img0 = cv2.applyColorMap(vis_img0, cv2.COLORMAP_JET)

    # filter
    outputs = F.max_pool2d(outputs0, kernel_size=(3, 3), stride=1, padding=1)
    keep = (outputs == outputs0).float()
    outputs = outputs * keep
    outputs = F.interpolate(outputs, scale_factor=8, mode="nearest")

    vis_img = outputs[0, 0].cpu().numpy()
    # normalize density map values from 0 to 1, then map it to 0-255.
    vis_img = (vis_img - vis_img.min()) / (
        vis_img.max() - vis_img.min() + 1e-5
    )
    vis_img = (vis_img * 255).astype(np.uint8)

    # vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
    # vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    return vis_img, int(count)


imagepath = "./input"  # input image folder
savepath = "./output"  # output image folder
countresultspath = "./output/DMCOUNT-results-orthophoto-raw-aerial-finetune-1022-customize-yolov8s-panet-aerial-finetune-1280-L2.txt"  # count result folder
if not os.path.exists(savepath):
    os.mkdir(savepath)

files = os.listdir(imagepath)
for file in files:
    if "jpg" in file:
        name = file.split(".")[0]
        img0 = cv2.imread(os.path.join(imagepath, file))
        img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        result, count = predict(img)

        result = cv2.resize(result, (img.shape[1], img.shape[0]))
        non_zero_mask = result != 0
        overlay = img0.copy()
        overlay[non_zero_mask] = (0, 0, 255)

        imgpath = os.path.join(savepath, file)
        # cv2.imwrite(imgpath, result)
        cv2.imwrite(imgpath, overlay)
        print("{},{}".format(name, count))
        with open(countresultspath, "a") as f:
            f.write("{},{} \n".format(name, count))
