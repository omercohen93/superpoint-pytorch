import torch
import kornia as K
import torch.nn.functional as F



def detector_loss(true_map, chi, v_mask=None):
  DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
  n, c, h, w = true_map.size()
  block_size = 8
  true_map = true_map.type(torch.float)
  unfolded_map = torch.nn.functional.unfold(true_map, block_size, stride=block_size)
  unfolded_map = unfolded_map.view(n, c * block_size ** 2, h // block_size, w // block_size)
  unfolded_map = unfolded_map.permute(0,2,3,1)
  shape = torch.cat([torch.tensor(unfolded_map.size())[:3], torch.tensor([1])], dim=0)
  unfolded_map = torch.cat([2*unfolded_map, torch.ones(tuple(shape)).to(DEVICE)], dim=3)
  noise = torch.rand(unfolded_map.size())*0.1
  noise = noise.to(DEVICE)
  label = torch.argmax(unfolded_map+noise,dim=3)
  #define valid mask
  if not v_mask is None:
    valid_mask = v_mask.type(torch.float32).to(DEVICE)
  else:
    valid_mask = torch.ones_like(true_map, dtype=torch.float32).to(DEVICE)  
  # adjust valid_mask
  valid_mask = F.unfold(valid_mask, block_size, stride=block_size)
  valid_mask = valid_mask.view(n, c * block_size ** 2, h // block_size, w // block_size)
  valid_mask = valid_mask.permute(0,2,3,1)
  valid_mask = torch.prod(valid_mask, dim=3)
  label[valid_mask==0] = 65
  #get loss
  m = torch.nn.LogSoftmax(dim=1)  
  loss = torch.nn.NLLLoss(ignore_index=65)
  output = loss(m(chi), label)
  return output





def descriptor_loss(DESC, warp_DESC, H, H_invert, v_mask):
  DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
  H_invert = H_invert.unsqueeze(1)
  desc = DESC.permute(0,2,3,1)
  B,Hc,Wc = tuple(desc.size())[:3]
  
  #create grid of center of HcxWc region
  cords = torch.stack(torch.meshgrid(torch.range(0,Hc-1),torch.range(0,Wc-1)), dim=-1).type(torch.int32).to(DEVICE)
  cords = cords.unsqueeze(0)
  cords = cords*8 + 4
  
  #change from ij to xy cords to warp grid
  xy_cords = torch.cat((cords[:,:,:,1].unsqueeze(3),cords[:,:,:,0].unsqueeze(3)),dim=-1)
  xy_warp_cords = K.geometry.warp.warp_grid(xy_cords,H_invert)
  
  #change back to ij
  warp_cords = torch.cat((xy_warp_cords[:,:,:,1].unsqueeze(3),xy_warp_cords[:,:,:,0].unsqueeze(3)),dim=-1)
  
  # calc S
  '''
  S[id_batch, h, w, h', w'] == 1 if the point of coordinates (h, w) warped by 
  the homography is at a distance from (h', w') less than 8 and 0 otherwise
  '''
  cords = cords.view((1,1,1,Hc,Wc,2)).type(torch.float)
  warp_cords = warp_cords.view((B, Hc, Wc, 1, 1, 2))
  distance_map = torch.norm(cords-warp_cords, dim=-1)
  S = distance_map <= 7.5
  S = S.type(torch.float)
  
  #descriptors
  desc = DESC.view((B, Hc, Wc, 1, 1, -1))
  desc = F.normalize(desc, dim=-1)
  warp_desc = warp_DESC.view((B, 1, 1, Hc, Wc, -1)) 
  warp_desc = F.normalize(warp_desc, dim=-1)
  
  #dot product calc
  ''' 
  dot_product_desc[id_batch, h, w, h', w'] is the dot product between the
  descriptor at position (h, w) in the original descriptors map and the
  descriptor at position (h', w') in the warped image
  '''
  dot_product = torch.sum(desc*warp_desc,dim=-1)
  relu = torch.nn.ReLU()
  dot_product = relu(dot_product)
  
  dot_product = F.normalize(dot_product.view((B, Hc, Wc, Hc * Wc)),dim=3)
  dot_product = dot_product.view((B, Hc, Wc, Hc, Wc)) 

  dot_product = F.normalize(dot_product.view((B, Hc * Wc, Hc, Wc)),dim=1)
  dot_product = dot_product.view((B, Hc, Wc, Hc, Wc)) 

  # Compute the loss
  pos_margin = 1
  neg_margin = 0.2
  lambda_d = 250
  positive_dist = torch.max(torch.zeros_like(dot_product), pos_margin - dot_product)
  negative_dist = torch.max(torch.zeros_like(dot_product), dot_product - neg_margin)
  loss = lambda_d * S * positive_dist + (1 - S) * negative_dist

  # adjust valid_mask
  block_size = 8
  valid_mask = F.unfold(v_mask, block_size, stride=block_size)
  valid_mask = valid_mask.view(B, block_size ** 2, Hc, Wc)
  valid_mask = valid_mask.permute(0,2,3,1)
  valid_mask = torch.prod(valid_mask, dim=3)
  valid_mask = valid_mask.view((B,1,1,Hc,Wc))
  
  normalization = torch.sum(valid_mask, dim=(1,2,3,4)) * Hc * Wc
  loss = torch.sum(loss*valid_mask, dim=(1,2,3,4))/normalization
  loss = torch.sum(loss)
  
  return loss
