import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

# Import model classes (make sure these files are on the import path)
from cross_atte_ori import MiniDA3_ViTBase
from networks import DepthAnythingV2FisheyeNet 

# =========================================================
# 1. Geometry generator (splatting)
# =========================================================
class SmoothDepthGenerator(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        # Intrinsics kept identical to the training script
        self.LEFT_PARAMS = {'f': 145.41, 'k1': -0.0239, 'cx_offset': -5.0, 'cy_offset': -5.0}
        self.RIGHT_PARAMS = {'f': 149.93, 'k1': -0.0313, 'cx_offset': 5.0, 'cy_offset': 5.0}
        
        ang = np.deg2rad(-75.0)
        c, s = np.cos(ang), np.sin(ang)
        mat_rot = torch.tensor([[c, 0, -s, 0], [0, 1, 0, 0], [s, 0, c, 0], [0, 0, 0, 1]], device=self.device).float()
        mat_trans = torch.eye(4, device=self.device).float(); mat_trans[0, 3] = 0.2
        M_step1 = torch.matmul(mat_trans, mat_rot)
        self.M_right_cam_to_world = torch.matmul(mat_rot, M_step1)
        
    def fisheye_grid_to_3d(self, h, w, param):
        v, u = torch.meshgrid(torch.arange(h, device=self.device), torch.arange(w, device=self.device), indexing='ij')
        cx, cy = w/2.0 + param['cx_offset'], h/2.0 + param['cy_offset']
        u_val, v_val = u - cx, -(v - cy)
        phi = torch.atan2(v_val, u_val)
        r_pix = torch.sqrt(u_val**2 + v_val**2)
        theta = r_pix / param['f'] / (1 + param['k1'] * (r_pix / param['f'])**2)
        mask = theta < np.deg2rad(110.1)
        sin_t, cos_t = torch.sin(theta), torch.cos(theta)
        vecs = torch.stack([sin_t*torch.cos(phi), sin_t*torch.sin(phi), cos_t], dim=-1)
        return vecs, mask

    def bilinear_splatting(self, points_3d, dist, ph, pw):
        lon = torch.atan2(points_3d[:,0], points_3d[:,2])
        lat = torch.asin(torch.clamp(points_3d[:,1] / (dist + 1e-8), -1.0, 1.0))
        u_pano = ((lon / (2 * np.pi)) + 0.5) * (pw - 1)
        v_pano = ((-lat / np.pi) + 0.5) * (ph - 1)
        
        u0, v0 = torch.floor(u_pano).long(), torch.floor(v_pano).long()
        u1, v1 = torch.clamp(u0 + 1, 0, pw - 1), torch.clamp(v0 + 1, 0, ph - 1)
        u0, v0 = torch.clamp(u0, 0, pw - 1), torch.clamp(v0, 0, ph - 1)
        
        wu, wv = u_pano - u0.float(), v_pano - v0.float()
        depth_weight = 1.0 / (dist + 0.1)
        depth_accum = torch.zeros((ph, pw), device=self.device)
        weight_accum = torch.zeros((ph, pw), device=self.device)
        
        for (ux, vx, w) in [(u0, v0, (1-wu)*(1-wv)), (u0, v1, (1-wu)*wv), (u1, v0, wu*(1-wv)), (u1, v1, wu*wv)]:
            idx = vx * pw + ux
            combined_w = w * depth_weight
            depth_accum.view(-1).scatter_add_(0, idx, dist * combined_w)
            weight_accum.view(-1).scatter_add_(0, idx, combined_w)
            
        return torch.where(weight_accum > 1e-6, depth_accum / weight_accum, torch.tensor(80.0, device=self.device))

    def generate_pano_depth(self, dep_l, dep_r, ph, pw):
        vl, ml = self.fisheye_grid_to_3d(*dep_l.shape, self.LEFT_PARAMS)
        pts_w_l = vl * dep_l.unsqueeze(-1)
        
        vr, mr = self.fisheye_grid_to_3d(*dep_r.shape, self.RIGHT_PARAMS)
        pts_local_r = vr * dep_r.unsqueeze(-1)
        ones = torch.ones_like(dep_r).unsqueeze(-1)
        pts_w_r = torch.matmul(torch.cat([pts_local_r, ones], dim=-1), self.M_right_cam_to_world.t())[..., :3]
        
        all_pts = torch.cat([pts_w_l[ml & (dep_l <= 80.0)], pts_w_r[mr & (dep_r <= 80.0)]], dim=0)
        dist = torch.linalg.norm(all_pts, dim=1)
        return self.bilinear_splatting(all_pts, dist, ph, pw)

# =========================================================
# 2. Visualization helpers
# =========================================================
def depth_to_jet_clipped(depth, vmin=0.0, vmax=80.0):
    d_viz = depth.copy().astype(np.float32)
    d_norm = np.clip((d_viz - vmin) / (vmax - vmin + 1e-8), 0, 1)
    d_uint8 = (d_norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(d_uint8, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    mask_invalid = (d_viz <= 0.0)
    color[mask_invalid] = 0
    return color

# =========================================================
# 3. Inference entry point
# =========================================================
def test_inference():
    # Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_DEPTH = 80.0
    IMG_SIZE = 518
    num = 1473
    # Model checkpoints
    PRETRAINED_FISH = "/media/doc-lab/HDCL-UT/super_yh/model_nogt_outdoor/model/upproj/wo_grad/checkpoint-2.pth.tar"
    CHECKPOINT_MINIDA3 = "/media/doc-lab/HDCL-UT/super_yh/attention/outdoor_res/checkpoints/minida3_ep54.pth"

    # Test sample paths
    paths = {
        'l_rgb': "/media/doc-lab/HDCL-UT/super_yh/test_data_1000/outdoor_1000/left_fish/"+str(num)+"_rgb.png",
        'r_rgb': "/media/doc-lab/HDCL-UT/super_yh/test_data_1000/outdoor_1000/right_fish/"+str(num)+"_rgb.png",
        'l_depth': "/media/doc-lab/HDCL-UT/super_yh/test_data_1000/outdoor_1000/left_fish/"+str(num)+"_depth.npy",
        'r_depth':"/media/doc-lab/HDCL-UT/super_yh/test_data_1000/outdoor_1000/right_fish/"+str(num)+"_depth.npy",
        'pano_depth': "/media/doc-lab/HDCL-UT/super_yh/test_data_1000/outdoor_1000/eqr/loc010_00"+str(num)+"_depth.npy"
    }

    # --- 1. Load models ---
    print(">>> Loading models...")
    model_fish = DepthAnythingV2FisheyeNet(
        model_size='vitb', pretrained_path=PRETRAINED_FISH, 
        scene_type='outdoor', max_depth=MAX_DEPTH, output_img_size=IMG_SIZE
    ).to(DEVICE).eval()

    model_test = MiniDA3_ViTBase(max_depth=MAX_DEPTH).to(DEVICE).eval()
    model_test.load_state_dict(torch.load(CHECKPOINT_MINIDA3, map_location=DEVICE))
    
    pano_gen = SmoothDepthGenerator(device=DEVICE)

    # --- 2. Load and preprocess data ---
    print(">>> Processing input data...")
    def load_img(path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    def load_depth(path, w, h):
        if not os.path.exists(path):
            print(f"Warning: {path} does not exist")
            return np.zeros((h, w), dtype=np.float32)
        d = np.load(path).astype(np.float32)
        d[d > MAX_DEPTH] = MAX_DEPTH  # Clamp to MAX_DEPTH (80 m)
        return cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)

    input_l_rgb = load_img(paths['l_rgb']).to(DEVICE)
    input_r_rgb = load_img(paths['r_rgb']).to(DEVICE)
    gt_l_depth = load_depth(paths['l_depth'], IMG_SIZE, IMG_SIZE)
    gt_r_depth = load_depth(paths['r_depth'], IMG_SIZE, IMG_SIZE)
    gt_p_depth = load_depth(paths['pano_depth'], IMG_SIZE * 2, IMG_SIZE)

    # --- 3. Run inference ---
    print(">>> Running inference...")
    with torch.no_grad():
        # A. Fisheye depth prediction
        preds_fish = model_fish(input_l_rgb, input_r_rgb)
        pl = preds_fish['front_depth'] 
        pr = preds_fish['back_depth']

        # B. Geometric splatting
        # Note: pl[0,0] yields an [H, W] tensor
        pano_input_geom = pano_gen.generate_pano_depth(pl[0,0], pr[0,0], IMG_SIZE, IMG_SIZE * 2)

        # C. MiniDA3 refinement
        # Build a [1, 3, H, W] input and normalize
        x_in = (pano_input_geom.unsqueeze(0).unsqueeze(0) / MAX_DEPTH).repeat(1, 3, 1, 1)
        pred_pano = model_test(x_in)

    # --- 4. Assemble the visualization ---
    print(">>> Stitching results...")
    # Row 1: left + right fisheye RGB (518x1036)
    rgb_row = np.hstack([
        (input_l_rgb[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8),
        (input_r_rgb[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    ])

    # Row 2: predicted fisheye depth (518x1036)
    v_pl = depth_to_jet_clipped(pl[0,0].cpu().numpy())
    v_pr = depth_to_jet_clipped(pr[0,0].cpu().numpy())
    pred_fish_row = np.hstack([v_pl, v_pr])

    # Row 3: ground-truth fisheye depth (518x1036)
    v_gt_l = depth_to_jet_clipped(gt_l_depth)
    v_gt_r = depth_to_jet_clipped(gt_r_depth)
    gt_fish_row = np.hstack([v_gt_l, v_gt_r])

    # Row 4: predicted panoramic depth (518x1036)
    v_pred_pano = depth_to_jet_clipped(pred_pano[0,0].cpu().numpy())
    
    # Row 5: ground-truth panoramic depth (518x1036)
    v_gt_pano = depth_to_jet_clipped(gt_p_depth)

    # Stack all rows vertically
    full_viz = np.vstack([
        rgb_row,        # 1. RGB
        pred_fish_row,  # 2. Fish Pred
        gt_fish_row,    # 3. Fish GT
        v_pred_pano,    # 4. Pano Pred (MiniDA3)
        v_gt_pano       # 5. Pano GT
    ])

    # Optional text labels
    labels = [
        "1. Fish RGB (Left + Right)",
        "2. Predicted Fish Depth (0-80m)",
        "3. Ground Truth Fish Depth (0-80m)",
        "4. MiniDA3 Predicted Panorama (0-80m)",
        "5. Ground Truth Panorama (0-80m)"
    ]
    #for i, txt in enumerate(labels):
        #cv2.putText(full_viz, txt, (15, 40 + i * IMG_SIZE), 
                    #cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Save result
    save_name = "/home/doc-lab/デスクトップ/test/test_result_34769.png"
    Image.fromarray(full_viz).save(save_name)
    print(f">>> Done. Full result saved to: {save_name}")

if __name__ == '__main__':
    test_inference()
