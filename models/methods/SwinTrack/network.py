import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class SwinTrack(nn.Module):
    def __init__(self, backbone, encoder, decoder, out_norm, head,
                 z_backbone_out_stage, x_backbone_out_stage,
                 z_input_projection, x_input_projection,
                 z_pos_enc, x_pos_enc):
        super(SwinTrack, self).__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.out_norm = out_norm
        self.head = head

        self.z_backbone_out_stage = z_backbone_out_stage
        self.x_backbone_out_stage = x_backbone_out_stage
        self.z_input_projection = z_input_projection
        self.x_input_projection = x_input_projection

        self.z_pos_enc = z_pos_enc
        self.x_pos_enc = x_pos_enc

        self.reset_parameters()

    def reset_parameters(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if self.z_input_projection is not None:
            self.z_input_projection.apply(_init_weights)
        if self.x_input_projection is not None:
            self.x_input_projection.apply(_init_weights)

        self.encoder.apply(_init_weights)
        self.decoder.apply(_init_weights)

    def initialize(self, z):
        return self._get_template_feat(z)

    def track(self, z_feat, x):
        x_feat = self._get_search_feat(x)

        return self._track(z_feat, x_feat)

    def forward(self, z, x, z_feat=None):
        """
        Combined entry point for training and inference (include initialization and tracking).
            Args:
                z (torch.Tensor | None)
                x (torch.Tensor | None)
                z_feat (torch.Tensor | None)

            Training:
                Input:
                    z: (B, H_z * W_z, 3), template image
                    x: (B, H_x * W_x, 3), search image
                Return:
                    Dict: Output of the head, like {'class_score': torch.Tensor(B, num_classes, H, W), 'bbox': torch.Tensor(B, H, W, 4)}.
            Inference:
                Initialization:
                    Input:
                        z: (B, H_z * W_z, 3)
                    Return:
                        torch.Tensor: (B, H_z * W_z, dim)
                Tracking:
                    Input:
                        z_feat: (B, H_z * W_z, dim)
                        x: (B, H_x * W_x, 3)
                    Return:
                        Dict: Same as training.
            """
        if z_feat is None:
            z_feat = self.initialize(z)
        if x is not None:
            return self.track(z_feat, x)
        else:
            return z_feat

    def _get_template_feat(self, z):
        z_feat, = self.backbone(z, (self.z_backbone_out_stage,), False)
        if self.z_input_projection is not None:
            z_feat = self.z_input_projection(z_feat)
        return z_feat

    def _get_search_feat(self, x):
        x_feat, = self.backbone(x, (self.x_backbone_out_stage,), False)
        if self.x_input_projection is not None:
            x_feat = self.x_input_projection(x_feat)
        return x_feat

    def _track(self, z_feat, x_feat):
        z_pos = None
        x_pos = None

        if self.z_pos_enc is not None:
            z_pos = self.z_pos_enc().unsqueeze(0)
        if self.x_pos_enc is not None:
            x_pos = self.x_pos_enc().unsqueeze(0)

        z_feat, x_feat = self.encoder(z_feat, x_feat, z_pos, x_pos)

        decoder_feat = self.decoder(z_feat, x_feat, z_pos, x_pos)
        decoder_feat = self.out_norm(decoder_feat)

        return self.head(decoder_feat)
