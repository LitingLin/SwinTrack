def build_backbone(config: dict, load_pretrained=True):
    backbone_config = config['backbone']
    if 'parameters' in backbone_config:
        backbone_build_params = backbone_config['parameters']
        if load_pretrained and 'pretrained' in backbone_build_params:
            load_pretrained = backbone_build_params['pretrained']
            del backbone_build_params['pretrained']
    else:
        backbone_build_params = ()
    if backbone_config['type'] == 'swin_transformer':
        from models.backbone.swin_transformer import build_swin_transformer_backbone
        if 'embed_dim' in backbone_build_params:
            backbone_build_params['overwrite_embed_dim'] = backbone_build_params['embed_dim']
            del backbone_build_params['embed_dim']
        backbone = build_swin_transformer_backbone(load_pretrained=load_pretrained, **backbone_build_params)
    elif backbone_config['type'] == 'resnet50' and backbone_config['mod'] == 'layer_3_dilation_2':
        from models.backbone.resnet50_mod_layer_3_dilation_2 import resnet50
        backbone = resnet50(pretrained=load_pretrained, **backbone_build_params)
    else:
        raise Exception(f'unsupported {backbone_config["type"]}')

    return backbone
