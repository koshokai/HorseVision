import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class MiniDA3_ViTBase(nn.Module):
    def __init__(self, max_depth=80.0):
        super().__init__()
        self.max_depth = max_depth
        self.backbone = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, dynamic_img_size=True)
        self.blocks = self.backbone.blocks
        self.stages = [2, 5, 8, 11]
        embed_dim = 768

        self.reassemble = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, 256, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.GroupNorm(16, 256),
                nn.LeakyReLU(0.2, inplace=True)
            ) for _ in range(4)
        ])


        self.fusion_init = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.Sequential(
            nn.PixelShuffle(2), # 512 -> 128
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # smooth layer
            nn.ELU()
        )
        
        self.skip1 = nn.Conv2d(256, 128, kernel_size=1)
        self.refine1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2), # 256 -> 64
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # smooth layer
            nn.ELU()
        )
        
        self.up3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PixelShuffle(2), # 64 -> 16
            nn.Conv2d(16, 16, kernel_size=3, padding=1)
        )

        self.final_head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        B, C, H, W = x.shape
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        feat_h, feat_w = H // 14, W // 14

        layer_features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.stages:
                f = x[:, 1:, :].transpose(1, 2).reshape(B, 768, feat_h, feat_w)
                layer_features.append(f)

        out_feats = [self.reassemble[i](f) for i, f in enumerate(layer_features)]
        
        # decoder
        combined = torch.cat(out_feats, dim=1)
        d = self.fusion_init(combined)
        
        # fusion skip line feature
        d = self.up1(d)
        s1 = F.interpolate(self.skip1(out_feats[1]), size=d.shape[-2:], mode='bilinear')
        d = self.refine1(torch.cat([d, s1], dim=1))
        
        d = self.up2(d)
        d = self.up3(d)
        
        out = self.final_head(d)
        final = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
        
        # scale recover(0~1 -> 0~20m)
        return torch.clamp(final * self.max_depth, min=0.1, max=self.max_depth)
