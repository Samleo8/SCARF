import cv2
import os
import torch
import numpy as np
import imageio
import pickle as pkl
import random


def image_projection_wo_dist_mat(XX, c2w):
    XXc = np.matmul(XX, c2w[:3, :3]) -  np.matmul(c2w[:3, 3], c2w[:3, :3])
    pts = XXc[:, :2] / XXc[:, 2][:, np.newaxis]

    pts = pts * fc + cc
    return pts

def multi_world2cam_grid_sample_mat(XX, c2w):
    pts = image_projection_wo_dist_mat(XX, c2w)
    pts = np.array(pts) / np.array([nx//2, ny//2]) - 1
    return pts

def image_projection_wo_dist_mat_torch(XX, c2w, device):
    XXc = torch.matmul(XX, c2w[:3,:3]) - torch.matmul(c2w[:3,3], c2w[:3,:3])
    pts = XXc[:,:2] / XXc[:,2].unsqueeze(-1)

    pts = pts * torch.Tensor(fc).to(device) + torch.Tensor(cc).to(device)
    return pts

def multi_world2cam_grid_sample_mat_torch(XX, c2w, device):
    pts = image_projection_wo_dist_mat_torch(XX, c2w, device)
    pts = pts / torch.tensor([nx//2, ny//2]).to(device) - 1
    return pts

def image_projection_wo_dist(XX, pose):
    Rc, _ = cv2.Rodrigues(np.array(pose_extrinsics['omc_{}'.format(int(pose))]))
    Tc = np.array(pose_extrinsics['Tc_{}'.format(int(pose))])

    XXc = np.matmul(XX, Rc.T) + Tc
    pts = XXc[:, :2] / XXc[:, 2][:, np.newaxis]

    pts = pts * fc + cc
    # print(Rc,Tc, fc, cc)
    return pts

def multi_world2cam_grid_sample(XX, pose):
    pts = image_projection_wo_dist(XX, pose)
    pts = np.array(pts) / np.array([nx//2, ny//2]) - 1
    return pts

def img_string(scan, pose, lighting):
    if isinstance(lighting, int):
        return f"{scan}/rect_{pose}_{lighting}_r5000.png"
    else:
        return f"{scan}/rect_{pose}_{lighting}.png"

def load_scan_data(scan, mode, num_views, cfg, specific_poses = None, fixed = True, shuffle = False):

    if mode == 'train':
        poses_idx = random.sample(images[scan],num_views)
    elif specific_poses is None:
        if fixed:
            poses_idx = images[scan][cfg.fixed_batch * num_views:(cfg.fixed_batch + 1) * num_views]
        else:
            poses_idx = random.sample(images[scan], num_views)
    else:
        if not len(specific_poses) == num_views:
            raise ValueError('Poses are invalid.')
        poses_idx = images[scan][specific_poses]

    if shuffle:
        random.shuffle(poses_idx)

    imgs = []
    poses = []
    for pose in poses_idx:
        # always use max lightning images - these have the minimal amount of pose dependent shadows
        img_name = img_string(scan, pose, 'max')
        fname = os.path.join(basedir, img_name)
        img = imageio.imread(fname)

        if cfg.half_res:
            img_half_res = np.zeros(( ny, nx, 3))
            img_half_res[:] = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_AREA)
            img = img_half_res
        imgs.append(img)
        poses.append(pose_extrinsics[f'c2w_{int(pose)}'])

    imgs = (np.array(imgs) / 255.).astype(np.float32)
    poses = np.array(poses).astype(np.float32)

    return imgs, poses, poses_idx

def setup_DTU(mode, cfg):
    load_parameters()

    global basedir, split, val_samples, images, light_cond
    basedir = cfg.datadir
    split, images = pkl.load(open( basedir + f'/{cfg.split}', 'rb'))


    global ny, nx, fc, cc
    if cfg.half_res:
        nx = nx//2
        ny = ny//2
        fc = fc/2.
        cc = cc/2


    return split[mode], ny, nx, fc, cc, 'x_down_y_down_z_cam_dir'



def load_parameters():
    global pose_extrinsics
    pose_extrinsics = {  # Image #1:
        # Image #1:
        'omc_1': [2.646883e+00, 8.491224e-01, 1.201473e+00],
        'Tc_1': [-9.164054e+01, 2.883847e+01, 5.780082e+02],

        # Image #2:
        'omc_2': [-2.444423e+00, -1.383376e+00, -9.331597e-01],
        'Tc_2': [-9.677179e+01, -1.515467e+01, 5.991584e+02],

        # Image #3:
        'omc_3': [-1.973200e+00, -1.703606e+00, -5.772585e-01],
        'Tc_3': [-7.798528e+01, -5.694933e+01, 6.165702e+02],

        # Image #4:
        'omc_4': [-1.459378e+00, -1.890948e+00, -2.412967e-01],
        'Tc_4': [-3.987781e+01, -8.548137e+01, 6.237389e+02],

        # Image #5:
        'omc_5': [-9.261897e-01, -1.962692e+00, 6.906567e-02],
        'Tc_5': [8.307069e+00, -9.358140e+01, 6.197556e+02],

        # Image #6:
        'omc_6': [-9.871599e-01, -1.988992e+00, 3.035277e-01],
        'Tc_6': [1.361858e+01, -8.935096e+01, 6.383336e+02],

        # Image #7:
        'omc_7': [-1.353674e+00, -1.994332e+00, 8.911246e-02],
        'Tc_7': [-1.893242e+01, -8.745466e+01, 6.418942e+02],

        # Image #8:
        'omc_8': [-1.718859e+00, -1.948508e+00, -1.471893e-01],
        'Tc_8': [-5.010895e+01, -7.627709e+01, 6.388482e+02],

        # Image #9:
        'omc_9': [-2.075648e+00, -1.844694e+00, -4.067469e-01],
        'Tc_9': [-7.538925e+01, -5.719246e+01, 6.289479e+02],

        # Image #10:
        'omc_10': [-2.411937e+00, -1.674094e+00, -6.890162e-01],
        'Tc_10': [-9.206173e+01, -3.200799e+01, 6.124841e+02],

        # Image #11:
        'omc_11': [2.578526e+00, 1.354978e+00, 9.332071e-01],
        'Tc_11': [-9.732819e+01, -4.681446e+00, 5.930120e+02],

        # Image #12:
        'omc_12': [2.325060e+00, 1.037853e+00, 9.201125e-01],
        'Tc_12': [-9.226156e+01, 1.073483e+01, 5.735922e+02],

        # Image #13:
        'omc_13': [2.411389e+00, 1.354611e+00, 7.815021e-01],
        'Tc_13': [-9.728733e+01, -6.581466e+00, 5.927909e+02],

        # Image #14:
        'omc_14': [2.463799e+00, 1.678324e+00, 6.141950e-01],
        'Tc_14': [-9.457204e+01, -2.482871e+01, 6.119319e+02],

        # Image #15:
        'omc_15': [-2.365436e+00, -1.916759e+00, -3.953420e-01],
        'Tc_15': [-8.473506e+01, -4.236416e+01, 6.298787e+02],

        # Image #16:
        'omc_16': [-2.092482e+00, -1.999459e+00, -1.531067e-01],
        'Tc_16': [-6.939086e+01, -5.774461e+01, 6.447952e+02],

        # Image #17:
        'omc_17': [-1.808456e+00, -2.040524e+00, 6.387093e-02],
        'Tc_17': [-4.797623e+01, -6.984002e+01, 6.543738e+02],

        # Image #18:
        'omc_18': [-1.520611e+00, -2.047848e+00, 2.608970e-01],
        'Tc_18': [-2.335541e+01, -7.774938e+01, 6.592142e+02],

        # Image #19:
        'omc_19': [-1.231173e+00, -2.023609e+00, 4.399609e-01],
        'Tc_19': [2.900203e+00, -8.088525e+01, 6.587835e+02],

        # Image #20:
        'omc_20': [-1.176651e+00, -1.994369e+00, 7.478962e-01],
        'Tc_20': [1.790292e+01, -6.953225e+01, 6.727010e+02],

        # Image #21:
        'omc_21': [-1.420318e+00, -2.057120e+00, 6.066878e-01],
        'Tc_21': [-3.945601e+00, -6.924403e+01, 6.756760e+02],

        # Image #22:
        'omc_22': [-1.664645e+00, -2.100258e+00, 4.483199e-01],
        'Tc_22': [-2.614640e+01, -6.620822e+01, 6.740619e+02],

        # Image #23:
        'omc_23': [-1.907884e+00, -2.120815e+00, 2.723920e-01],
        'Tc_23': [-4.679229e+01, -6.066039e+01, 6.681046e+02],

        # Image #24:
        'omc_24': [-2.148903e+00, -2.116922e+00, 7.960078e-02],
        'Tc_24': [-6.472668e+01, -5.283662e+01, 6.579810e+02],

        # Image #25:
        'omc_25': [2.340362e+00, 2.046854e+00, 1.403923e-01],
        'Tc_25': [-7.953085e+01, -4.303474e+01, 6.437174e+02],

        # Image #26:
        'omc_26': [2.327473e+00, 1.800362e+00, 3.396890e-01],
        'Tc_26': [-9.097947e+01, -3.182349e+01, 6.271888e+02],

        # Image #27:
        'omc_27': [2.286973e+00, 1.553755e+00, 5.151899e-01],
        'Tc_27': [-9.620540e+01, -1.989732e+01, 6.087574e+02],

        # Image #28:
        'omc_28': [2.223241e+00, 1.307865e+00, 6.715248e-01],
        'Tc_28': [-9.612917e+01, -8.135497e+00, 5.896122e+02],

        # Image #29:
        'omc_29': [1.981169e+00, 1.166382e+00, 6.532234e-01],
        'Tc_29': [-9.224942e+01, -8.170887e+00, 5.760951e+02],

        # Image #30:
        'omc_30': [2.058507e+00, 1.363190e+00, 5.073172e-01],
        'Tc_30': [-9.650260e+01, -1.467603e+01, 5.950254e+02],

        # Image #31:
        'omc_31': [2.121523e+00, 1.559759e+00, 3.467072e-01],
        'Tc_31': [-9.599625e+01, -2.166255e+01, 6.141893e+02],

        # Image #32:
        'omc_32': [2.168215e+00, 1.755050e+00, 1.708726e-01],
        'Tc_32': [-9.112517e+01, -2.877758e+01, 6.326956e+02],

        # Image #33:
        'omc_33': [2.195775e+00, 1.947120e+00, -2.096002e-02],
        'Tc_33': [-8.197386e+01, -3.543869e+01, 6.495122e+02],

        # Image #34:
        'omc_34': [2.202600e+00, 2.135357e+00, -2.273745e-01],
        'Tc_34': [-6.945205e+01, -4.144506e+01, 6.641077e+02],

        # Image #35:
        'omc_35': [-2.081689e+00, -2.206824e+00, 4.280003e-01],
        'Tc_35': [-5.368839e+01, -4.664949e+01, 6.759346e+02],

        # Image #36:
        'omc_36': [-1.864397e+00, -2.167204e+00, 5.999613e-01],
        'Tc_36': [-3.546530e+01, -5.070710e+01, 6.841887e+02],

        # Image #37:
        'omc_37': [-1.646364e+00, -2.109852e+00, 7.530470e-01],
        'Tc_37': [-1.564667e+01, -5.353292e+01, 6.884525e+02],

        # Image #38:
        'omc_38': [-1.428464e+00, -2.036978e+00, 8.876757e-01],
        'Tc_38': [4.615616e+00, -5.495444e+01, 6.886891e+02],

        # Image #39:
        'omc_39': [-1.346521e+00, -1.931540e+00, 1.213192e+00],
        'Tc_39': [2.105140e+01, -3.765155e+01, 6.966033e+02],

        # Image #40:
        'omc_40': [-1.546048e+00, -2.034804e+00, 1.107969e+00],
        'Tc_40': [2.678049e+00, -3.703716e+01, 6.991479e+02],

        # Image #41:
        'omc_41': [-1.746760e+00, -2.128177e+00, 9.839430e-01],
        'Tc_41': [-1.605455e+01, -3.639403e+01, 6.980469e+02],

        # Image #42:
        'omc_42': [-1.948536e+00, -2.209766e+00, 8.406961e-01],
        'Tc_42': [-3.427551e+01, -3.496254e+01, 6.931018e+02],

        # Image #43:
        'omc_43': [2.065551e+00, 2.187158e+00, -6.483938e-01],
        'Tc_43': [-5.234873e+01, -3.295325e+01, 6.845457e+02],

        # Image #44:
        'omc_44': [2.064156e+00, 2.045547e+00, -4.299055e-01],
        'Tc_44': [-6.710444e+01, -3.083513e+01, 6.727221e+02],

        # Image #45:
        'omc_45': [2.045593e+00, 1.898180e+00, -2.245764e-01],
        'Tc_45': [-7.912639e+01, -2.837521e+01, 6.582456e+02],

        # Image #46:
        'omc_46': [2.011775e+00, 1.745960e+00, -3.156369e-02],
        'Tc_46': [-8.808723e+01, -2.563881e+01, 6.417008e+02],

        # Image #47:
        'omc_47': [1.963131e+00, 1.587510e+00, 1.511683e-01],
        'Tc_47': [-9.430976e+01, -2.284552e+01, 6.235811e+02],

        # Image #48:
        'omc_48': [1.901970e+00, 1.427657e+00, 3.195682e-01],
        'Tc_48': [-9.544957e+01, -2.008661e+01, 6.045579e+02],

        # Image #49:
        'omc_49': [1.829100e+00, 1.265243e+00, 4.765918e-01],
        'Tc_49': [-9.289060e+01, -1.753753e+01, 5.856687e+02],

        # Image #50:
        'omc_50': [2.305701e+00, 1.649061e+00, 4.492638e-01],
        'Tc_50': [-6.383454e+01, -2.000424e+01, 8.588008e+02],

        # Image #51:
        'omc_51': [2.336728e+00, 1.913978e+00, 2.520153e-01],
        'Tc_51': [-5.612380e+01, -3.250388e+01, 8.776873e+02],

        # Image #52:
        'omc_52': [-2.261703e+00, -2.107734e+00, -2.144868e-02],
        'Tc_52': [-4.239672e+01, -4.379922e+01, 8.947052e+02],

        # Image #53:
        'omc_53': [-2.003101e+00, -2.122963e+00, 1.968417e-01],
        'Tc_53': [-2.444751e+01, -5.339777e+01, 9.072840e+02],

        # Image #54:
        'omc_54': [-1.743182e+00, -2.110398e+00, 3.920183e-01],
        'Tc_54': [-2.692817e+00, -6.040591e+01, 9.152671e+02],

        # Image #55:
        'omc_55': [-1.800575e+00, -2.153530e+00, 6.669151e-01],
        'Tc_55': [1.277855e+00, -4.653850e+01, 9.297923e+02],

        # Image #56:
        'omc_56': [-2.031932e+00, -2.203248e+00, 4.904495e-01],
        'Tc_56': [-1.780457e+01, -4.233753e+01, 9.220189e+02],

        # Image #57:
        'omc_57': [2.192155e+00, 2.163407e+00, -2.827982e-01],
        'Tc_57': [-3.529691e+01, -3.703861e+01, 9.100701e+02],

        # Image #58:
        'omc_58': [2.187616e+00, 1.965779e+00, -6.000276e-02],
        'Tc_58': [-4.924033e+01, -3.092801e+01, 8.948635e+02],

        # Image #59:
        'omc_59': [2.159061e+00, 1.763380e+00, 1.471817e-01],
        'Tc_59': [-5.908918e+01, -2.423857e+01, 8.770318e+02],

        # Image #60:
        'omc_60': [1.952343e+00, 1.624750e+00, 8.440668e-02],
        'Tc_60': [-6.308337e+01, -1.882429e+01, 8.719969e+02],

        # Image #61:
        'omc_61': [2.001443e+00, 1.788918e+00, -1.156076e-01],
        'Tc_61': [-5.453175e+01, -2.152204e+01, 8.909074e+02],

        # Image #62:
        'omc_62': [2.033262e+00, 1.945662e+00, -3.277532e-01],
        'Tc_62': [-4.392913e+01, -2.404548e+01, 9.079835e+02],

        # Image #63:
        'omc_63': [2.047557e+00, 2.096818e+00, -5.533052e-01],
        'Tc_63': [-2.965379e+01, -2.666796e+01, 9.224436e+02],

        # Image #64:
        'omc_64': [2.041834e+00, 2.239327e+00, -7.953803e-01],
        'Tc_64': [-1.217832e+01, -2.829657e+01, 9.336400e+02],

        # Image #65:
        'omc_65': [-2.241562e+00, -1.871111e+00, -4.116853e-01],
        'Tc_65': [-1.084174e+02, -5.221257e+01, 4.233987e+02],

        # Image #66:
        'omc_66': [-2.091573e+00, -2.088623e+00, 4.327380e-02],
        'Tc_66': [-9.028940e+01, -6.036205e+01, 4.500333e+02],

        # Image #67:
        'omc_67': [2.210867e+00, 2.002388e+00, -5.989890e-02],
        'Tc_67': [-1.053918e+02, -4.170273e+01, 4.468390e+02],

        # Image #68:
        'omc_68': [-2.186611e+00, -2.211694e+00, 3.142882e-01],
        'Tc_68': [-8.843018e+01, -4.904300e+01, 4.633488e+02],

        # Image #69:
        'omc_69': [2.049481e+00, 2.099737e+00, -5.568376e-01],
        'Tc_69': [-8.731116e+01, -3.493689e+01, 4.737004e+02],

        # Image #70:
        'omc_70': [2.032290e+00, 1.928660e+00, -3.030334e-01],
        'Tc_70': [-1.030307e+02, -3.236226e+01, 4.571523e+02],

        # Image #71:
        'omc_71': [-2.606700e+00, -1.223481e+00, -1.069454e+00],
        'Tc_71': [-1.132880e+02, -5.469785e-01, 4.702208e+02],

        # Image #72:
        'omc_72': [-2.037239e+00, -1.670249e+00, -6.221543e-01],
        'Tc_72': [-9.738356e+01, -5.455919e+01, 4.935668e+02],

        # Image #73:
        'omc_73': [-1.394966e+00, -1.905683e+00, -2.018047e-01],
        'Tc_73': [-4.965851e+01, -9.003543e+01, 5.025802e+02],

        # Image #74:
        'omc_74': [-1.342353e+00, -2.008646e+00, 1.651027e-01],
        'Tc_74': [-3.068429e+01, -8.843448e+01, 5.253814e+02],

        # Image #75:
        'omc_75': [-1.768932e+00, -1.967846e+00, -1.137668e-01],
        'Tc_75': [-6.728917e+01, -7.596682e+01, 5.211497e+02],

        # Image #76:
        'omc_76': [-2.182247e+00, -1.848268e+00, -4.268242e-01],
        'Tc_76': [-9.539345e+01, -5.336013e+01, 5.072734e+02],

        # Image #77:
        'omc_77': [-2.565417e+00, -1.633148e+00, -7.739022e-01],
        'Tc_77': [-1.111033e+02, -2.401960e+01, 4.852666e+02],

        # Image #78:
        'omc_78': [2.341683e+00, 1.420819e+00, 6.745048e-01],
        'Tc_78': [-1.128133e+02, -1.428001e+01, 4.764629e+02],

        # Image #79:
        'omc_79': [2.398385e+00, 1.772606e+00, 4.606326e-01],
        'Tc_79': [-1.079589e+02, -3.262772e+01, 4.997512e+02],

        # Image #80:
        'omc_80': [-2.293659e+00, -2.027653e+00, -1.919011e-01],
        'Tc_80': [-9.368268e+01, -4.992736e+01, 5.209539e+02],

        # Image #81:
        'omc_81': [-1.972716e+00, -2.076011e+00, 8.032695e-02],
        'Tc_81': [-7.165014e+01, -6.415126e+01, 5.364520e+02],

        # Image #82:
        'omc_82': [-1.645013e+00, -2.077783e+00, 3.187288e-01],
        'Tc_82': [-4.462654e+01, -7.374743e+01, 5.450456e+02],

        # Image #83:
        'omc_83': [-1.566009e+00, -2.088817e+00, 7.313139e-01],
        'Tc_83': [-2.605532e+01, -6.035093e+01, 5.646674e+02],

        # Image #84:
        'omc_84': [-1.845229e+00, -2.154476e+00, 5.397172e-01],
        'Tc_84': [-5.076234e+01, -5.656479e+01, 5.598666e+02],

        # Image #85:
        'omc_85': [-2.122968e+00, -2.191486e+00, 3.186463e-01],
        'Tc_85': [-7.366487e+01, -5.058596e+01, 5.490755e+02],

        # Image #86:
        'omc_86': [2.237895e+00, 2.049452e+00, -6.200746e-02],
        'Tc_86': [-9.202590e+01, -4.266297e+01, 5.329912e+02],

        # Image #87:
        'omc_87': [2.214639e+00, 1.795809e+00, 1.852566e-01],
        'Tc_87': [-1.053908e+02, -3.326140e+01, 5.126827e+02],

        # Image #88:
        'omc_88': [2.158746e+00, 1.537841e+00, 4.052071e-01],
        'Tc_88': [-1.116148e+02, -2.307911e+01, 4.895776e+02],

        # Image #89:
        'omc_89': [1.918138e+00, 1.467809e+00, 2.788666e-01],
        'Tc_89': [-1.117462e+02, -2.307711e+01, 4.878987e+02],

        # Image #90:
        'omc_90': [1.988559e+00, 1.666634e+00, 6.257752e-02],
        'Tc_90': [-1.084421e+02, -2.660926e+01, 5.114138e+02],

        # Image #91:
        'omc_91': [2.037990e+00, 1.860626e+00, -1.742962e-01],
        'Tc_91': [-9.808955e+01, -3.013093e+01, 5.329242e+02],

        # Image #92:
        'omc_92': [2.063431e+00, 2.045867e+00, -4.281304e-01],
        'Tc_92': [-8.365366e+01, -3.348138e+01, 5.514660e+02],

        # Image #93:
        'omc_93': [2.062645e+00, 2.223165e+00, -7.053591e-01],
        'Tc_93': [-6.371835e+01, -3.624792e+01, 5.657904e+02],

        # Image #94:
        'omc_94': [-1.847898e+00, -2.171450e+00, 9.136035e-01],
        'Tc_94': [-4.199641e+01, -3.827475e+01, 5.748073e+02],

    }
    for i in range(1, 95):
        pose = i
        w2c = cv2.Rodrigues(np.array(pose_extrinsics['omc_{}'.format(pose)]))[0]
        c2w = w2c.T
        translation = np.array([pose_extrinsics['Tc_{}'.format(pose)]])
        c2w = np.hstack((c2w, np.matmul(c2w, - translation.T)))
        c2w = np.vstack((c2w, [0, 0, 0, 1]))
        pose_extrinsics[f'c2w_{pose}'] = c2w

    # Focal length:
    global fc
    fc = np.array([2892.843725329502400, 2882.249450476587300])

    # Principal point:
    global cc
    cc = np.array([824.425157504919530, 605.187152104484080])

    # Image size:
    global nx, ny
    nx = 1600
    ny = 1200

def load_cam_path():
    global camera_path
    poses = [27,15,23,44,27]
    camera_path = []

    for i, pose in enumerate(poses[:-1]):
        frames = 14
        for interpol in np.linspace(0,1,frames)[:frames - 1]:
            pose = (1-interpol) * pose_extrinsics[f'c2w_{poses[i]}'] + interpol * pose_extrinsics[f'c2w_{poses[i+1]}']
            camera_path.append(pose)
    return camera_path

def load_cam_path_debug():
    global camera_path
    poses = [27,15]
    camera_path = []

    for i, pose in enumerate(poses[:-1]):
        frames = 14
        for interpol in np.linspace(1/14,0.5,frames)[:frames - 1]:
            pose = (1-interpol) * pose_extrinsics[f'c2w_{poses[i]}'] + interpol * pose_extrinsics[f'c2w_{poses[i+1]}']
            camera_path.append(pose)
    return camera_path


def load_pose(pose):
    return pose_extrinsics[f'c2w_{pose}']