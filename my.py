import torch
import numpy as np
import json

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

if __name__ == "__main__":
    v = [[19, 77], [21.5, 60], [20, 46], [23, 27]]
    f = dict()
    f['camera_angle_x'] = 0.6911112070083618
    f['frames'] = []
    a = 0
    for i in range(0, 4):
        for j in range(0, 720, 45):
            tmp = {}
            tmp["file_path"] = "./train/r_%d" % a
            a += 1
            tmp["rotation"] = 0.012566370614359171
            tmp["transform_matrix"] = pose_spherical(j/2, v[i][1], v[i][0]/100).tolist()
            f['frames'].append(tmp)
    print(json.dumps(f, indent=4))
    with open('mydata/transforms_train.json', 'w') as fil:
        json.dump(f, fil, indent=4)