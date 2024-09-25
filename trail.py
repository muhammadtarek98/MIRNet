from collections import OrderedDict
import torch, torchvision, cv2, kornia
import numpy as np
from networks.MIRNet_model import MIRNet
from pixmamba.mmagic.tests.configs.two_stage_test import model


def load_model(device, ckpt, model):
    new_state_dict = OrderedDict()
    state_dict = torch.load(f=ckpt)
    for k, v in state_dict["state_dict"].items():
        new_k = k[7:]
        new_state_dict[new_k] = v
    model.load_state_dict(state_dict=new_state_dict)
    return model


def img_2_tensor(img):
    transform = torchvision.transforms.Compose(
        transforms=[
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(size=(480, 480),
                                          interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT),
            torchvision.transforms.Normalize(mean=[0, 0, 0],
                                             std=[1, 1, 1])])
    raw_image_tensor = transform(img)
    raw_image_tensor = torch.unsqueeze(input=raw_image_tensor, dim=0)
    return raw_image_tensor


def tensor_2_img(tensor: torch.Tensor):
    tensor = torch.squeeze(tensor)
    tensor = torch.permute(input=tensor, dims=(1, 2, 0))
    tensor = tensor.detach().cpu().numpy()
    arr = np.clip(a=tensor, a_min=0, a_max=1)
    arr = (arr * 255.0).astype(np.uint8)
    return arr


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = "/home/cplus/projects/m.tarek_master/Image_enhancement/weights/MIRNet/model_denoising.pth"
model = load_model(device=device, ckpt=ckpt, model=MIRNet())
model.eval()
image = cv2.imread("/home/cplus/projects/m.tarek_master/Image_enhancement/bad_visibility_images/264_cam1left_2016_10_21_07_10_15.jpg")
raw_tensor=img_2_tensor(img=image)
with torch.no_grad():
    pred=model(raw_tensor)

output=tensor_2_img(tensor=pred)
print(output.shape)
cv2.imshow(winname="test",mat=output)
cv2.waitKey(0)
cv2.destroyAllWindows()