import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ==================== Automatically import the official DepthAnythingV2 ====================

def setup_depth_anything_v2():
    """Automatically search for and import the official DepthAnythingV2"""
    current_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    
    possible_paths = [
        Path('set/your/path/Depth-Anything-V2'),
        current_dir / 'Depth-Anything-V2',
    ]
    
    dav2_path = None
    for path in possible_paths:
        if path.exists() and (path / 'depth_anything_v2').exists():
            dav2_path = path
            print(f"[INFO] Found Depth-Anything-V2: {dav2_path}")
            break
    
    if dav2_path is None:
        raise FileNotFoundError(
            f"Depth-Anything-V2 directory not found!!\n"
            f"Please make sure Depth-Anything-V2 is located in one of the following paths:\n" +
            "\n".join([f"  - {p}" for p in possible_paths])
        )
    
    sys.path.insert(0, str(dav2_path))
    
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
        print("[INFO] ✓ DepthAnythingV2 imported successfully!\n")
        return DepthAnythingV2, dav2_path
    except ImportError as e:
        raise ImportError(f"Failed to import DepthAnythingV2: {e}")

DepthAnythingV2, DAV2_BASE_PATH = setup_depth_anything_v2()



def create_fisheye_mask(images, threshold=1e-5, kernel_size=5):
    
    B, C, H, W = images.shape
    
    if C == 3:
        gray = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
    else:
        gray = images[:, 0]
    
    img_max = images.max().item()
    if img_max <= 1.0:
        effective_threshold = threshold / 255.0
    else:
        effective_threshold = threshold
    
    mask = (gray > effective_threshold).float()
    
    if mask.sum() == 0:
        mask = (gray > 0.001).float()
    
    if kernel_size > 0:
        mask = mask.unsqueeze(1)
        # dilation
        mask = F.max_pool2d(mask, kernel_size, stride=1, padding=kernel_size//2)
        # erosion
        mask = -F.max_pool2d(-mask, kernel_size, stride=1, padding=kernel_size//2)
        return mask
    
    return mask.unsqueeze(1)

# ==================== CrossViewFusion ====================

class CrossViewFusion(nn.Module):
    
    def __init__(self, fusion_type='weighted'):
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'weighted':

            self.alpha = nn.Parameter(torch.tensor(0.7))
            self.beta = nn.Parameter(torch.tensor(0.3))
            
        elif fusion_type == 'attention':
            
            self.conv_q = nn.Conv2d(1, 16, 3, padding=1)
            self.conv_k = nn.Conv2d(1, 16, 3, padding=1)
            self.conv_v = nn.Conv2d(1, 16, 3, padding=1)
            self.conv_out = nn.Conv2d(16, 1, 1)
            self.scale = 16 ** 0.5
        
        elif fusion_type == 'conv':
            
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(2, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 2, 1)
            )
    
    def forward(self, depth1, depth2):
        if self.fusion_type == 'none':
            return depth1, depth2
            
        elif self.fusion_type == 'weighted':
            
            alpha = torch.sigmoid(self.alpha)
            beta = torch.sigmoid(self.beta)
            depth1_fused = alpha * depth1 + (1 - alpha) * depth2
            depth2_fused = beta * depth2 + (1 - beta) * depth1
            return depth1_fused, depth2_fused
            
        elif self.fusion_type == 'attention':
            B, _, H, W = depth1.shape
            
            
            q1 = self.conv_q(depth1).flatten(2).transpose(1, 2)  # [B, HW, 16]
            k2 = self.conv_k(depth2).flatten(2)  # [B, 16, HW]
            v2 = self.conv_v(depth2).flatten(2).transpose(1, 2)  # [B, HW, 16]
            
            attn = torch.bmm(q1, k2) / self.scale  # [B, HW, HW]
            attn = attn.softmax(dim=-1)
            fused_flat = torch.bmm(attn, v2)  # [B, HW, 16]
            fused = fused_flat.transpose(1, 2).view(B, 16, H, W)
            depth1_fused = depth1 + self.conv_out(fused)
            
            # View2 attend to View1
            q2 = self.conv_q(depth2).flatten(2).transpose(1, 2)
            k1 = self.conv_k(depth1).flatten(2)
            v1 = self.conv_v(depth1).flatten(2).transpose(1, 2)
            
            attn2 = torch.bmm(q2, k1) / self.scale
            attn2 = attn2.softmax(dim=-1)
            fused2_flat = torch.bmm(attn2, v1)
            fused2 = fused2_flat.transpose(1, 2).view(B, 16, H, W)
            depth2_fused = depth2 + self.conv_out(fused2)
            
            return depth1_fused, depth2_fused
        
        elif self.fusion_type == 'conv':
            
            concat = torch.cat([depth1, depth2], dim=1)  # [B, 2, H, W]
            fused = self.fusion_conv(concat)  # [B, 2, H, W]
            depth1_fused = fused[:, 0:1]
            depth2_fused = fused[:, 1:2]
            return depth1_fused, depth2_fused
        
        return depth1, depth2

# ==================== Main Network ====================

class DepthAnythingV2FisheyeNet(nn.Module):
    
    MODEL_CONFIGS = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    def __init__(self,
                 model_size='vitb',
                 pretrained_path='/your/model/path',
                 scene_type='indoor',          # 'indoor'(20m) or 'outdoor'(80m)
                 max_depth=None,               
                 input_size=518,               # DAv2 input size
                 output_img_size=512,          # DAV2 output size
                 shared_encoder=True,          
                 use_fusion=False,             
                 fusion_type='weighted'):      # 'none'|'weighted'|'attention'|'conv'
        
        super().__init__()
        
        self.model_size = model_size
        self.input_size = input_size
        self.output_img_size = output_img_size
        self.shared_encoder = shared_encoder
        self.use_fusion = use_fusion
        self.scene_type = scene_type
        
        # Setting max depth
        if max_depth is None:
            self.max_depth = 20.0 if scene_type == 'indoor' else 80.0
        else:
            self.max_depth = max_depth
        
        # Eval setting
        if model_size not in self.MODEL_CONFIGS:
            raise ValueError(f"model_size must be one of {list(self.MODEL_CONFIGS.keys())}")
        
        # Getting model configuration
        config = self.MODEL_CONFIGS[model_size].copy()
        
        # Metric depth models require the max_depth parameter
        config['max_depth'] = self.max_depth
        
        # Creating encoder
        mode_str = f"matric depth({scene_type}, max_depth={self.max_depth}m)"
        
        if shared_encoder:
            print(f"[INFO] Creating shared Depth Anything V2 {model_size.upper()} {mode_str} encoder...")
            self.depth_model = DepthAnythingV2(**config)
            self.depth_model_front = self.depth_model
            self.depth_model_back = self.depth_model
        else:
            print(f"[INFO] Creating independent Depth Anything V2 {model_size.upper()} {mode_str} dual encoders...")
            self.depth_model_front = DepthAnythingV2(**config)
            self.depth_model_back = DepthAnythingV2(**config)
            self.depth_model = self.depth_model_front
        
        # Loading pretrained weights
        self._load_pretrained_weights(pretrained_path)
        
        # cross fusion
        if use_fusion:
            print(f"[INFO] add module (fusion_type={fusion_type})")
            self.fusion = CrossViewFusion(fusion_type=fusion_type)
        
        # ImageNet Norm
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        
        self._print_parameter_stats()
    
    def _find_pretrained_checkpoint(self, pretrained_path):
        """find pretrain model weight"""
        if pretrained_path is not None:
            path = Path(pretrained_path)
            if path.exists():
                print(f"[INFO] Using the specified pretrained weights: {path}")
                return path
        
        checkpoint_dir = DAV2_BASE_PATH / 'checkpoints'
        
        # matric models
        dataset = 'hypersim' if self.scene_type == 'indoor' else 'vkitti'
        
        # search model type（include .pth and .tar）
        possible_names = [
            f'depth_anything_v2_metric_{dataset}_{self.model_size}.tar',
            f'depth_anything_v2_metric_{dataset}_{self.model_size}.pth',
            f'depth_anything_v2_metric_{dataset}_{self.model_size}.pt',
            f'depth_anything_v2_metric_{dataset}_{self.model_size}.pth.tar',
        ]
        print(f"[INFO] Searching for metric depth pretrained weights ({self.scene_type})...")
        
        for name in possible_names:
            path = checkpoint_dir / name
            if path.exists():
                print(f"[INFO] Pretrained weights found: {path}")
                return path
        
        print(f"[WARNING] ⚠️ Pretrained weights not found!")
        print(f"[Hint] Expected file names: {possible_names}")
        print(f"[Hint] Expected directory: {checkpoint_dir}")
        print(f"[Hint] Randomly initialized weights will be used")
        return None
    
    def _load_pretrained_weights(self, pretrained_path):
        
        checkpoint_path = self._find_pretrained_checkpoint(pretrained_path)
        if checkpoint_path is None:
            return        
        try:
            print(f"[INFO] loading pretrain model weight...")
            loaded_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if isinstance(loaded_data, dict):
                if 'depth_net' in loaded_data:
                    depth_net = loaded_data['depth_net']
                    if isinstance(depth_net, nn.Module):
                        if hasattr(depth_net, 'max_depth'):
                            tar_max_depth = depth_net.max_depth
                            if tar_max_depth != self.max_depth:
                                #print(f"[WARNING] ⚠️ max_depth mismatch!")
                                #print(f"            Target model: {tar_max_depth}m")
                                #print(f"            Current model: {self.max_depth}m")
                                #print(f"            This will cause different depth prediction scales!")
                                print(f"            Suggestion: set max_depth={tar_max_depth} when creating the model")
                        
                        state_dict = depth_net.state_dict()
                        
                    elif isinstance(depth_net, dict):
                        state_dict = depth_net
                    else:
                        raise TypeError(f"depth_net type is not supported: {type(depth_net)}")
                    
                elif 'state_dict' in loaded_data:
                    state_dict = loaded_data['state_dict']
                elif 'model' in loaded_data:
                    model_data = loaded_data['model']
                    if isinstance(model_data, dict):
                        state_dict = model_data
                    elif isinstance(model_data, nn.Module):
                        state_dict = model_data.state_dict()
                    else:
                        state_dict = model_data
                elif 'model_state_dict' in loaded_data:
                    state_dict = loaded_data['model_state_dict']
                else:

                    state_dict = loaded_data
            elif isinstance(loaded_data, nn.Module):
                print(f"[INFO] Detected model object format, extracting state_dict...")
                state_dict = loaded_data.state_dict()
            else:
                state_dict = loaded_data

            # 3. state_dict
            if not isinstance(state_dict, dict):
                raise TypeError(f"The extracted state_dict is not a dictionary type: {type(state_dict)}")

            has_front = any(k.startswith('depth_model_front.') for k in state_dict.keys())
            has_back = any(k.startswith('depth_model_back.') for k in state_dict.keys())
            has_base = any(k.startswith('depth_model.') for k in state_dict.keys())
            
            if (has_front or has_back) and self.shared_encoder:                
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('depth_model_front.'):
                        new_key = k.replace('depth_model_front.', 'depth_model.')
                        new_state_dict[new_key] = v
                    elif not k.startswith('depth_model_back.'):
                        new_state_dict[k] = v                
                state_dict = new_state_dict            
            elif has_base and not (has_front or has_back) and not self.shared_encoder:
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_state_dict[k] = v
                    if k.startswith('depth_model.'):
                        base_key = k[len('depth_model.'):]
                        new_state_dict[f'depth_model_front.{base_key}'] = v.clone()
                        new_state_dict[f'depth_model_back.{base_key}'] = v.clone()
                
                state_dict = new_state_dict
            else:
                print(f"[INFO] Parameter structure matches, loading directly")

            new_state_dict = {}
            has_module_prefix = False
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    has_module_prefix = True
                    new_state_dict[k[7:]] = v  
                else:
                    new_state_dict[k] = v
            
            if has_module_prefix:
                print(f"[INFO] Detected and removed 'module.' prefix")
            
            state_dict = new_state_dict

            # 6. loading
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            
            if self.shared_encoder:
                print(f"[INFO] ✓ Pretrained weights loaded successfully! (shared encoder)")
            else:
                print(f"[INFO] ✓ Pretrained weights loaded successfully! (independent encoders)")
             
        except Exception as e:
            print(f"[ERROR] ❌ Failed to load weights: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _print_parameter_stats(self):

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        mode = "Shared Encoder" if self.shared_encoder else "Indeoendent Dual Encoder"
        
        #print(f"\n{'='*70}")
        #print(f"model: Depth Anything V2 {self.model_size.upper()} - metric depth ({mode})")
        #print(f"{'='*70}")
        #print(f"total paramaters:       {total/1e6:.2f}M")
        #print(f"Trainable parameters:   {trainable/1e6:.2f}M ({trainable/total*100:.1f}%)")
        #print(f"Input size:     {self.input_size}×{self.input_size}")
        #print(f"Output size     {self.output_img_size}×{self.output_img_size}")
        #print(f"Depth range:     0 - {self.max_depth}m ({self.scene_type})")
        #print(f"{'='*70}\n")
    
    def _prepare_input_tensor(self, image_tensor):

        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
        
        resized = F.interpolate(
            image_tensor, 
            size=(self.input_size, self.input_size), 
            mode='bilinear', 
            align_corners=False
        )
        
        normalized = (resized - self.mean) / self.std
        
        return normalized
    
    def _infer_depth(self, model, input_tensor):

        depth = model(input_tensor)
        
        # 4D tensor [B, 1, H, W]
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        elif depth.dim() == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        
        return depth
    
    def forward(self, front_image, back_image, auto_mask=True, mask_threshold=0.01):

        device = front_image.device
        B = front_image.shape[0]
        
        # 1. Creating mask
        if auto_mask:
            front_mask_orig = create_fisheye_mask(front_image, threshold=mask_threshold)
            back_mask_orig = create_fisheye_mask(back_image, threshold=mask_threshold)
        
        # 2. 518×518 + ImageNet Norm
        front_input = self._prepare_input_tensor(front_image)
        back_input = self._prepare_input_tensor(back_image)
        
        # 3. ✅ DAv2 inference
        front_depth = self._infer_depth(self.depth_model_front, front_input)
        back_depth = self._infer_depth(self.depth_model_back, back_input)
        
        # 4. size
        front_depth = F.interpolate(
            front_depth, 
            size=(self.output_img_size, self.output_img_size),
            mode='bilinear', 
            align_corners=False
        )
        back_depth = F.interpolate(
            back_depth,
            size=(self.output_img_size, self.output_img_size),
            mode='bilinear',
            align_corners=False
        )
        
        # 5. None
        if self.use_fusion:
            front_depth, back_depth = self.fusion(front_depth, back_depth)
        
        result = {
            'front_depth': front_depth,
            'back_depth': back_depth
        }
        
        # 6. Applying mask
        if auto_mask:
            front_mask = F.interpolate(
                front_mask_orig, 
                size=(self.output_img_size, self.output_img_size),
                mode='bilinear', 
                align_corners=False
            )
            back_mask = F.interpolate(
                back_mask_orig,
                size=(self.output_img_size, self.output_img_size),
                mode='bilinear',
                align_corners=False
            )
            
            result['front_depth'] = front_depth * front_mask
            result['back_depth'] = back_depth * back_mask
            result['front_mask'] = front_mask
            result['back_mask'] = back_mask
        
        return result
