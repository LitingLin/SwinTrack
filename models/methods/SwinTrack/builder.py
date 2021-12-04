from core.run.event_dispatcher.register import EventRegister
from models.utils.drop_path import DropPathAllocator, DropPathScheduler
import torch.nn as nn
from .modules.encoder.builder import build_encoder
from .modules.decoder.builder import build_decoder
from .positional_encoding.builder import build_position_embedding
from models.backbone.builder import build_backbone
from models.head.builder import build_head
from .network import SwinTrack
from data.tracking.methods.SiamFC.pseudo_data import build_siamfc_pseudo_data_generator


def build_swin_track_main_components(config, num_epochs, iterations_per_epoch, event_register: EventRegister, has_training_run):
    transformer_config = config['transformer']

    drop_path_config = transformer_config['drop_path']
    drop_path_allocator = DropPathAllocator(drop_path_config['rate'])

    backbone_dim = transformer_config['backbone']['dim']
    transformer_dim = transformer_config['dim']

    z_shape = transformer_config['backbone']['template']['shape']
    x_shape = transformer_config['backbone']['search']['shape']
    backbone_out_stage = transformer_config['backbone']['stage']

    z_input_projection = None
    x_input_projection = None
    if backbone_dim != transformer_dim:
        z_input_projection = nn.Linear(backbone_dim, transformer_dim)
        x_input_projection = nn.Linear(backbone_dim, transformer_dim)

    num_heads = transformer_config['num_heads']
    mlp_ratio = transformer_config['mlp_ratio']
    qkv_bias = transformer_config['qkv_bias']
    drop_rate = transformer_config['drop_rate']
    attn_drop_rate = transformer_config['attn_drop_rate']

    position_embedding_config = transformer_config['position_embedding']
    z_pos_enc, x_pos_enc = build_position_embedding(position_embedding_config, z_shape, x_shape, transformer_dim)

    with drop_path_allocator:
        encoder = build_encoder(config, drop_path_allocator,
                                transformer_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                                z_shape, x_shape)

        decoder = build_decoder(config, drop_path_allocator,
                                transformer_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                                z_shape, x_shape)

    out_norm = nn.LayerNorm(transformer_dim)

    if 'warmup' in drop_path_config and len(drop_path_allocator) > 0:
        if has_training_run:
            from models.utils.build_warmup_scheduler import build_warmup_scheduler
            scheduler = build_warmup_scheduler(drop_path_config['warmup'], drop_path_config['rate'],
                                               iterations_per_epoch, num_epochs)

            drop_path_scheduler = DropPathScheduler(drop_path_allocator.get_all_allocated(), scheduler)

            event_register.register_iteration_end_hook(drop_path_scheduler)
            event_register.register_epoch_begin_hook(drop_path_scheduler)

    return encoder, decoder, out_norm, backbone_out_stage, backbone_out_stage, z_input_projection, x_input_projection, z_pos_enc, x_pos_enc


def build_swin_track(config, load_pretrained, num_epochs, iterations_per_epoch, event_register: EventRegister, has_training_run):
    backbone = build_backbone(config, load_pretrained)
    encoder, decoder, out_norm, z_backbone_out_stage, x_backbone_out_stage, z_input_projection, x_input_projection, z_pos_enc, x_pos_enc = \
        build_swin_track_main_components(config, num_epochs, iterations_per_epoch, event_register, has_training_run)
    head = build_head(config)

    return SwinTrack(backbone, encoder, decoder, out_norm, head, z_backbone_out_stage, x_backbone_out_stage, z_input_projection, x_input_projection, z_pos_enc, x_pos_enc), \
           build_siamfc_pseudo_data_generator(config, event_register)
