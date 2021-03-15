import PIL
import torch
from torch import nn
import numpy as np
from utils.points import nms, get_prob_map
import kornia as K

class SuperPointNet(torch.nn.Module):
  def __init__(self, superpoint_bool):
    super(SuperPointNet, self).__init__()
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
    # Shared Encoder.
    self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
    self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
    self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
    self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
    self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
    self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
    self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
    self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
    # Detector Head.
    self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
    # Descriptor Head.
    self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convDb = nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
    self.superpoint_bool = superpoint_bool


  def forward(self, x):
    """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8(super) 
      or None(magicpoint).
    """
    # Shared Encoder.
    Encoder = nn.Sequential(self.conv1a, self.relu,
                            self.conv1b, self.relu,
                            self.pool,
                            self.conv2a, self.relu,
                            self.conv2b, self.relu,
                            self.pool,
                            self.conv3a, self.relu,
                            self.conv3b, self.relu,
                            self.pool,
                            self.conv4a, self.relu,
                            self.conv4b, self.relu)
    x = Encoder(x)                
    

    # Detector Head.
    cPa = self.relu(self.convPa(x))
    semi = self.convPb(cPa)


    # Descriptor Head.
    #if we want superpoint model:
    if self.superpoint_bool:
      cDa = self.relu(self.convDa(x))
      desc = self.convDb(cDa)
      dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
      desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
      
    #if we want magicpoint model:
    else:
      desc = None
    
    return semi, desc
    

def get_descriptors(desc):
  desc_full_size = torch.nn.functional.interpolate(desc,(desc.size()[2]*8,desc.size()[3]*8), mode='bilinear')
  dn = torch.norm(desc_full_size, p=2, dim=1) # Compute the norm.
  descriptor_output = desc_full_size.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
  return descriptor_output



import torchvision
def superpoint_frontend(im, model,N_nms=4,num_points=200):
  DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
  im_tens = np.asarray(im)
  transform = transforms.Compose([torchvision.transforms.ToTensor()]) 
  im = transform(im_tens).to(DEVICE)

  if len(im.shape)<4:
    im = im.unsqueeze(0)
  if len(im.shape)<4:
    im = im.unsqueeze(0).to(DEVICE)
  with torch.no_grad():
    semi, desc = model.forward(im)
  map = get_prob_map(semi)
  points = nms(map, N_nms,num_points)
  descriptors = get_descriptors(desc)
  map_im = K.tensor_to_image(points)
  cords = np.where(map_im>0)
  y = np.reshape(cords[0],(1,1,-1,1))
  x = np.reshape(cords[1],(1,1,-1,1))

  descriptors = descriptors[0,:,y.flatten(),x.flatten()]
  descriptors = np.array(descriptors.permute(1,0).cpu())

  cords = np.concatenate((x,y),axis=3)
  cords = cords.squeeze(0).squeeze(0)
  return cords, descriptors