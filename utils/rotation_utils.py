import torch


def normalize_vector( v, dim =1, return_mag =False):
    v_mag = torch.sqrt(v.pow(2).sum(dim=dim, keepdim=True))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.expand_as(v)
    v = v/v_mag
    return v

def cross_product(u, v):
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
    out = torch.cat((i.unsqueeze(1), j.unsqueeze(1), k.unsqueeze(1)),1)#batch*3
    return out

def Ortho6d2Mat(x_raw, y_raw):
    y = normalize_vector(y_raw)
    z = cross_product(x_raw, y)
    z = normalize_vector(z)#batch*3
    x = cross_product(y,z)#batch*3

    x = x.unsqueeze(2)
    y = y.unsqueeze(2)
    z = z.unsqueeze(2)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix

