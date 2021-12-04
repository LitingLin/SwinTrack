def build_position_embedding(position_embedding_config, z_shape, x_shape, dim):
    if not position_embedding_config['enabled']:
        return None, None

    if position_embedding_config['type'] == 'sine':
        from .sine import SinePositionEmbedding

        with_branch_index = position_embedding_config['with_branch_index']

        return SinePositionEmbedding(dim, (z_shape[1], z_shape[0]), 0 if with_branch_index else None), \
               SinePositionEmbedding(dim, (x_shape[1], x_shape[0]), 1 if with_branch_index else None)
    elif position_embedding_config['type'] == 'learned':
        from .learned import Learned2DPositionalEncoder

        return Learned2DPositionalEncoder(dim, z_shape[0], z_shape[1]), \
               Learned2DPositionalEncoder(dim, x_shape[0], x_shape[1])
    else:
        raise ValueError('Unknown positional embedding type: {}'.format(position_embedding_config['type']))
