import torch
import numpy as np
from torchvision import  transforms
import PIL

# get point map from coordinates set [x,y,V] where V is the value
def cords_to_map(label,im_size, thres_point=0.015, device=True):
  if device:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
  else:
    DEVICE = 'cpu'
  map = torch.zeros(im_size).type(torch.double).to(DEVICE)
  if len(label.size())<4:
    label = label.unsqueeze(1)
  label = label.type(torch.double)
  vals = label[:,:,:,2].flatten()
  x = label[:,:,:,0].type(torch.long).flatten()
  y = label[:,:,:,1].type(torch.long).flatten()
  B = torch.arange(im_size[0], dtype=torch.long)[:, None].expand(im_size[0], label.size(2)).flatten().to(DEVICE)
  map[B[vals>0],0,y[vals>0],x[vals>0]]=vals[vals>0]
  one = torch.tensor(1).to(DEVICE)
  zero = torch.tensor(0).to(DEVICE)
  map = torch.where(map>thres_point, one, zero)
  return map


# get coordinates set [x,y,V] where V is the value from point map
def map_to_cords(BATCH_SIZE, iter, points,names):
  cord = torch.where(points>0)
  y = cord[2].cpu()
  y = np.array(y.unsqueeze(1))
  x = cord[3].cpu()
  x = np.array(x.unsqueeze(1))
  prob = points[cord].cpu()
  prob = np.array(prob.unsqueeze(1))
  full_cord = np.concatenate((x,y,prob),axis=1)
  full_cord = full_cord.reshape((BATCH_SIZE,-1))
  new_cords = np.concatenate((names[BATCH_SIZE*iter:BATCH_SIZE*(iter+1),:],full_cord), axis=1)
  return new_cords


#perform non maximum supresion, posible to get tok k point, and/or all points above thres
def nms(prob, size, topk=None, thres=None):
  orig_shape = prob.size()
  pool = torch.nn.MaxPool2d(size,size, int(size/2), return_indices=True)
  unpool = torch.nn.MaxUnpool2d(size, size, int(size/2))
  folded_points, ind = pool(prob)
  points = unpool(folded_points, ind, orig_shape)

  if not topk is None:
    # Reshape and calculate positions of top 10%
    points = points.view(points.size(0), points.size(1), -1)
    nb_pixels = points.size(2)
    ret = torch.topk(points, k=topk, dim=2)
    ret.indices.shape

    # Scatter to zero'd tensor
    res = torch.zeros_like(points)
    res.scatter_(2, ret.indices, ret.values)
    points = res.view(orig_shape)
  if not thres is None:
    points[points<thres] = 0
  return points

#perform depth to space on detector output of the superpoint model
def get_prob_map(semi):
  m = torch.nn.Softmax(1)
  prob = m(semi)
  prob = prob[:, :-1, :, :]
  detector_output = torch.nn.functional.pixel_shuffle(prob, 8)
  return detector_output


#fix descriptor output to image size
def get_descriptors(desc):
  interp = transforms.Resize((desc.size()[2]*8,desc.size()[3]*8), PIL.Image.BICUBIC)
  desc_full_size = interp(desc)
  dn = torch.norm(desc_full_size, p=2, dim=1) # Compute the norm.
  descriptor_output = desc_full_size.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
  return descriptor_output
