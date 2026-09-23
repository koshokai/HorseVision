import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import math
# ==========================================
# 1.Base Loss 
# ==========================================
def get_gaussian_kernel_2d(kernel_size=11, sigma=3.0, device='cuda'):
    coords = torch.arange(kernel_size).float().to(device)
    coords -= (kernel_size - 1) / 2.0
    g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1d /= g1d.sum()  
    g2d = g1d.unsqueeze(1) * g1d.unsqueeze(0) 
    return g2d.unsqueeze(0).unsqueeze(0)

def apply_gaussian_blur(input_tensor, kernel_size=11, sigma=3.0):

    device = input_tensor.device
    kernel = get_gaussian_kernel_2d(kernel_size, sigma, device)
    padding = kernel_size // 2
    return F.conv2d(input_tensor, kernel, padding=padding)
class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

def compute_smooth_loss(tgt_depth, tgt_img):
    def gradient_x(img): return img[:, :, :, :-1] - img[:, :, :, 1:]
    def gradient_y(img): return img[:, :, :-1, :] - img[:, :, 1:, :]
    
    depth_grad_x = gradient_x(tgt_depth)
    depth_grad_y = gradient_y(tgt_depth)
    image_grad_x = gradient_x(tgt_img)
    image_grad_y = gradient_y(tgt_img)
    
    mean_grad_x = torch.mean(torch.abs(image_grad_x), 1, keepdim=True)
    mean_grad_y = torch.mean(torch.abs(image_grad_y), 1, keepdim=True)
    
    weights_x = torch.exp(-mean_grad_x) 
    weights_y = torch.exp(-mean_grad_y)
    
    return (depth_grad_x * weights_x).abs().mean() + (depth_grad_y * weights_y).abs().mean()

def compute_gradient_loss(pred, target, mask=None):

    def gradient(x):

        grad_x = x[:, :, :, :-1] - x[:, :, :, 1:]
        grad_y = x[:, :, :-1, :] - x[:, :, 1:, :]
        return grad_x, grad_y

    grad_pred_x, grad_pred_y = gradient(pred)
    grad_tgt_x, grad_tgt_y = gradient(target)

    diff_x = torch.abs(grad_pred_x - grad_tgt_x)
    diff_y = torch.abs(grad_pred_y - grad_tgt_y)

    if mask is not None:

        mask_x = mask[:, :, :, :-1] * mask[:, :, :, 1:]
        mask_y = mask[:, :, :-1, :] * mask[:, :, 1:, :]
        loss = (diff_x * mask_x).mean() + (diff_y * mask_y).mean()
    else:
        loss = diff_x.mean() + diff_y.mean()

    return loss

# ==========================================
# 2.  Warping 
# ==========================================
class FisheyeInverseWarper(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        #intrinsic
        self.RIGHT_PARAMS = {'f': 149.93, 'k1': -0.0313, 'cx_offset': 5.0, 'cy_offset': 5.0}
        self.LEFT_PARAMS = {'f': 145.41, 'k1': -0.0239, 'cx_offset': -5.0, 'cy_offset': -5.0}

    def forward_poly_proj(self, theta, f, k1):
        return f * (theta + k1 * theta**3)

    def inverse_poly_proj(self, r, f, k1, max_iter=5):
        theta = r / f
        for _ in range(max_iter):
            f_theta = f * (theta + k1 * theta**3) - r
            f_prime = f * (1 + 3 * k1 * theta**2)
            theta = theta - f_theta / (f_prime + 1e-8)
        return theta

    def get_pixel_grid(self, H, W):
        y_range = torch.arange(H, device=self.device, dtype=torch.float32)
        x_range = torch.arange(W, device=self.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_range, x_range, indexing='ij')
        return xx, yy

    def rotate_points(self, points, angle_deg, axis='y'):
        B = points.shape[0]
        angle_rad = torch.tensor(np.deg2rad(angle_deg), device=self.device).float()
        c, s = torch.cos(angle_rad), torch.sin(angle_rad)
        if axis == 'y':
            R = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], device=self.device)
        return torch.bmm(R.unsqueeze(0).repeat(B, 1, 1), points)

    def create_circle_mask(self, H, W, params):
        xx, yy = self.get_pixel_grid(H, W)
        cx = W / 2.0 + params['cx_offset']
        cy = H / 2.0 + params['cy_offset']
        r_circle = W / 2.0
        dist = torch.sqrt((xx - cx)**2 + (yy - cy)**2)
        mask = (dist <= r_circle).float()
        return mask.unsqueeze(0).unsqueeze(0)

    def inverse_warp(self, target_depth_meter, source_rgb, target_eye='left'):
        B, _, H, W = target_depth_meter.shape
        img_size = H
        target_depth_mm = target_depth_meter * 1000.0

        if target_eye == 'left':
            tgt_params, src_params = self.LEFT_PARAMS, self.RIGHT_PARAMS
            inv_angle, inv_trans = 75.0, torch.tensor([200.0, 0.0, 0.0], device=self.device).view(1, 3, 1)
        else:
            tgt_params, src_params = self.RIGHT_PARAMS, self.LEFT_PARAMS
            inv_angle, inv_trans = -75.0, torch.tensor([-200.0, 0.0, 0.0], device=self.device).view(1, 3, 1)

        # --- 1. geometry ---
        xx, yy = self.get_pixel_grid(H, W)
        xx_f, yy_f = xx.expand(B, -1, -1).flatten(1), yy.expand(B, -1, -1).flatten(1)
        
        cx, cy = img_size/2.0 + tgt_params['cx_offset'], img_size/2.0 + tgt_params['cy_offset']
        x_img, y_img = cx - xx_f, cy - yy_f
        r_img = torch.sqrt(x_img**2 + y_img**2)
        theta = self.inverse_poly_proj(r_img, tgt_params['f'], tgt_params['k1'])
        phi = torch.atan2(y_img, x_img)
        
        # 3D
        depth_flat = target_depth_mm.flatten(1)
        X_tgt = depth_flat * torch.sin(theta) * torch.cos(phi)
        Y_tgt = depth_flat * torch.sin(theta) * torch.sin(phi)
        Z_tgt = depth_flat * torch.cos(theta)
        
        # Transform 
        points_tgt = torch.stack([X_tgt, Y_tgt, Z_tgt], dim=1)
        points_src = self.rotate_points(self.rotate_points(points_tgt, inv_angle) + inv_trans, inv_angle)
        X_src, Y_src, Z_src = points_src[:, 0], points_src[:, 1], points_src[:, 2]

        # Project back
        r_3d_src = torch.sqrt(X_src**2 + Y_src**2)
        theta_src = torch.atan2(r_3d_src, Z_src)
        phi_src = torch.atan2(Y_src, X_src)
        r_img_src = self.forward_poly_proj(theta_src, src_params['f'], src_params['k1'])
        
        cx_src, cy_src = img_size/2.0 + src_params['cx_offset'], img_size/2.0 + src_params['cy_offset']
        u_src = cx_src - r_img_src * torch.cos(phi_src)
        v_src = cy_src - r_img_src * torch.sin(phi_src)

        norm_u = 2.0 * u_src / (W - 1) - 1.0
        norm_v = 2.0 * v_src / (H - 1) - 1.0
        grid = torch.stack([norm_u.view(B, H, W), norm_v.view(B, H, W)], dim=-1)

        # --- 2. RGB ---
        warped_rgb = F.grid_sample(source_rgb, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        # --- 3. Mask (Static Logic) ---
        mask_src_static = self.create_circle_mask(H, W, src_params)
        mask_src_batch = mask_src_static.expand(B, -1, -1, -1)
        warped_mask = F.grid_sample(mask_src_batch, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return warped_rgb, warped_mask

# ==========================================
# 3. Loss (Static Mask)
# ==========================================
class FisheyeSelfSupervisedLoss(nn.Module):
    def __init__(self, teacher_model,lambda_smooth=0.0,  #None
                                        alpha_ssim=0.0,  #Learnable
                                        alpha_align=0.0, #Learnable
                                        lambda_grad=0.0, #Learnable
                                        device='cuda'):
        super().__init__()
        self.device = device
        # Warper 
        self.warper = FisheyeInverseWarper(device) 
        self.ssim = SSIM().to(device)
        
        self.lambda_smooth = lambda_smooth
        self.alpha_ssim = alpha_ssim
        self.lambda_align = alpha_align
        self.lambda_grad = lambda_grad

        self.teacher_model = teacher_model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()


        self.static_mask_left = None
        self.static_mask_right = None
        self.is_mask_initialized = False

    def init_static_mask(self, batch_size, height, width):

        if self.is_mask_initialized:
            return

        print(f"⚡ Init mask ({height}x{width})...")
        
        dummy_depth = torch.ones((1, 1, height, width)).to(self.device) * 1.5
        dummy_rgb = torch.zeros((1, 3, height, width)).to(self.device)
        
        with torch.no_grad():
            # --- Step A: Compute Overlap Mask ---
            _, mask_l = self.warper.inverse_warp(dummy_depth, dummy_rgb, 'left')
            _, mask_r = self.warper.inverse_warp(dummy_depth, dummy_rgb, 'right')
            
            # Erosion operation: shrink inward by 31 pixels to avoid distorted edges

            shrink = 31
            m_l_eroded = -F.max_pool2d(-mask_l, kernel_size=shrink, stride=1, padding=shrink//2)
            m_r_eroded = -F.max_pool2d(-mask_r, kernel_size=shrink, stride=1, padding=shrink//2)
            
            self.static_mask_left = torch.clamp(m_l_eroded, 0.0, 1.0).detach()
            self.static_mask_right = torch.clamp(m_r_eroded, 0.0, 1.0).detach()

            # --- Step B: Compute Imaging Circle Mask (FOV Mask) ---
            # Purpose: remove black background borders and eliminate ring artifacts
            xx, yy = self.warper.get_pixel_grid(height, width)
            cx, cy = width / 2.0, height / 2.0
            r_circle = (height / 2.0) - 10.0 # Reserve a 10-pixel margin
            
            dist = torch.sqrt((xx - cx)**2 + (yy - cy)**2)
            fov_m = (dist <= r_circle).float()
            self.fov_mask = fov_m.unsqueeze(0).unsqueeze(0).to(self.device)

        self.is_mask_initialized = True
        print(f"✅ Physical prior initialization completed.")
    
    def compute_scale_aware_distillation_loss(self, student_depth, teacher_depth, static_mask, photo_error):
        # Core logic:
        # Use the static mask to suppress stripe artifacts,
        # and use the FOV mask to remove background regions.
        b, _, h, w = student_depth.shape
        fov_m = self.fov_mask.expand(b, 1, h, w)

        # 1. Compute original confidence (based on photometric error)
        #  exp(-alpha * error)
        raw_conf = torch.exp(-photo_error * 10.0)

        # 2. [Key to solving stripe artifacts]: compute the confidence used for the ratio
        # Must be strictly restricted within the static_mask (overlapping region);
        # stripe artifacts in non-overlapping regions will be forcibly zeroed out.
        ratio_conf = raw_conf * static_mask
        
        with torch.no_grad():
            conf_sum_batch = ratio_conf.sum(dim=(2, 3), keepdim=True) + 1e-8
            
            # If the overlapping region is misaligned or occluded, fall back to the median value

            if conf_sum_batch.mean() < 50:
                valid = (static_mask > 0.5)
                ratio = torch.median(student_depth[valid]) / (torch.median(teacher_depth[valid]) + 1e-6)
            else:
                # Dynamically weighted ratio computation: synchronize the scale of the dual fisheye images

                weighted_s = (student_depth * ratio_conf).sum(dim=(2, 3), keepdim=True) / conf_sum_batch
                weighted_t = (teacher_depth * ratio_conf).sum(dim=(2, 3), keepdim=True) / conf_sum_batch
                ratio = weighted_s / (weighted_t + 1e-6)
            
            ratio = torch.clamp(ratio, 0.01, 10.0)
            target_depth = teacher_depth * ratio

        # 3. Compute the distillation loss and mask out the black borders using the FOV mask
        loss_map = F.smooth_l1_loss(student_depth, target_depth, reduction='none', beta=1.0)
        
        # [Key to removing ring artifacts]: compute the mean only within the fov_mask region,
        # so that the black background does not contribute gradients
        # The output must be a scalar to avoid backward() errors
        final_loss = (loss_map * fov_m).sum() / (fov_m.sum() + 1e-8)
        
        # For clearer visualization, multiply the confidence map by the fov_mask as well,
        # so that the background becomes black
        viz_confidence = raw_conf * fov_m * static_mask
        
        return final_loss, ratio, viz_confidence

    def compute_photometric_loss(self, synth_img, real_img, mask):
        # L1
        diff = torch.abs(synth_img - real_img)
        valid_pixels = mask.sum() * 3 + 1e-8
        l1_loss = (diff * mask).sum() / valid_pixels
        
        # SSIM
        ssim_map = self.ssim(synth_img, real_img)
        ssim_loss = ((1 - ssim_map) * mask).sum() / (mask.sum() * 3 + 1e-8)

        loss = self.alpha_ssim * ssim_loss + (1 - self.alpha_ssim) * l1_loss
        return loss, l1_loss, ssim_loss

    def forward(self, left_rgb, right_rgb, left_depth_pred, right_depth_pred):
        """
        Full forward pass: integrates APCA confidence and full FOV masking

        """
        b, c, h, w = left_rgb.shape

        # 1. Initialize static masks (including static_mask and fov_mask)
        if not self.is_mask_initialized:
            self.init_static_mask(b, h, w)

        # Obtain the full FOV mask and expand it to the current batch size
        # This mask determines which regions contain valid image content
        # and is used to eliminate black border interference
        fov_m = self.fov_mask.expand(b, 1, h, w)

        # 2. Photometric Loss)
        # Warp Left: 
        synth_left, _ = self.warper.inverse_warp(left_depth_pred, right_rgb, 'left')
        static_mask_l = self.static_mask_left.expand(b, 1, h, w)
        loss_l, l1_l, ssim_l = self.compute_photometric_loss(synth_left, left_rgb, static_mask_l)
        
        diff_l = torch.abs(synth_left - left_rgb).mean(dim=1, keepdim=True)

        # Warp Right: 
        synth_right, _ = self.warper.inverse_warp(right_depth_pred, left_rgb, 'right')
        static_mask_r = self.static_mask_right.expand(b, 1, h, w)
        loss_r, l1_r, ssim_r = self.compute_photometric_loss(synth_right, right_rgb, static_mask_r)
        
        diff_r = torch.abs(synth_right - right_rgb).mean(dim=1, keepdim=True)
        
        # 3. load teacher
        with torch.no_grad():
            teacher_out = self.teacher_model(left_rgb, right_rgb)
            teacher_l = teacher_out['front_depth']
            teacher_r = teacher_out['back_depth']


        smooth_l = compute_smooth_loss(left_depth_pred * fov_m, left_rgb)
        smooth_r = compute_smooth_loss(right_depth_pred * fov_m, right_rgb)
        loss_smooth = (smooth_l + smooth_r) / 2.0
        
        # 5. Gradient Loss

        loss_grad_l = compute_gradient_loss(left_depth_pred * fov_m, teacher_l * fov_m)
        loss_grad_r = compute_gradient_loss(right_depth_pred * fov_m, teacher_r * fov_m)
        loss_grad = (loss_grad_l + loss_grad_r) / 2.0
        
        # 6. FOV Global Distillation
        align_l, ratio_l, conf_map_l = self.compute_scale_aware_distillation_loss(
            left_depth_pred, teacher_l, static_mask_l, diff_l)
        align_r, ratio_r, conf_map_r = self.compute_scale_aware_distillation_loss(
            right_depth_pred, teacher_r, static_mask_r, diff_r)
        
        loss_align = align_l + align_r

        # 7. total loss
        total_loss = (loss_l + loss_r) + \
                     self.lambda_smooth * loss_smooth + \
                     self.lambda_align * loss_align + \
                     self.lambda_grad * loss_grad

        losses = {
            'total': total_loss,
            'photo_l2r': loss_r,
            'photo_r2l': loss_l,
            'smooth': loss_smooth,
            'align': loss_align,
            'grad': loss_grad,
            'ssim_l': ssim_l,
            'ssim_r': ssim_r,
            'ratio_l': ratio_l.mean(),
            'ratio_r': ratio_r.mean()
        }
        
        images = {
            'synth_left': synth_left, 
            'mask_left': static_mask_l, 
            'fov_mask': fov_m,             
            'conf_map_l': conf_map_l,      
            'synth_right': synth_right, 
            'mask_right': static_mask_r,
            'conf_map_r': conf_map_r
        }
        return losses, images
