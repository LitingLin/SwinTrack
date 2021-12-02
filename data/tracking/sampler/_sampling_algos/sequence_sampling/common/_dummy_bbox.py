import numpy as np
from core.math.random.truncated_normal import truncated_normal_dist_generate_random_number
from data.operator.bbox.spatial.np.xyxy2cxcywh import box_xyxy2cxcywh
from data.operator.bbox.spatial.np.cxcywh2xyxy import box_cxcywh2xyxy


def generate_dummy_bbox_xyxy(image_size, rng_engine: np.random.Generator, reference_bbox: np.ndarray=None):
    dummy_bbox = np.zeros(4, dtype=np.float_)
    if reference_bbox is not None:
        reference_bbox = box_xyxy2cxcywh(reference_bbox)
        dummy_bbox[0] = truncated_normal_dist_generate_random_number(mean=reference_bbox[0], sd=0.2, low=0., upp=1.,
                                                                     rng_engine=rng_engine)
        dummy_bbox[1] = truncated_normal_dist_generate_random_number(mean=reference_bbox[1], sd=0.2, low=0., upp=1.,
                                                                     rng_engine=rng_engine)
        dummy_bbox[2] = truncated_normal_dist_generate_random_number(mean=reference_bbox[2], sd=0.2, low=0.1, upp=0.9,
                                                                     rng_engine=rng_engine)
        dummy_bbox[3] = truncated_normal_dist_generate_random_number(mean=reference_bbox[3], sd=0.2, low=0.1, upp=0.9,
                                                                     rng_engine=rng_engine)
    else:
        dummy_bbox[0] = truncated_normal_dist_generate_random_number(mean=0.5, sd=0.3, low=0., upp=1.,
                                                                     rng_engine=rng_engine)
        dummy_bbox[1] = truncated_normal_dist_generate_random_number(mean=0.5, sd=0.3, low=0., upp=1.,
                                                                     rng_engine=rng_engine)
        dummy_bbox[2] = truncated_normal_dist_generate_random_number(mean=0.2, sd=0.3, low=0.1, upp=0.9,
                                                                     rng_engine=rng_engine)
        dummy_bbox[3] = truncated_normal_dist_generate_random_number(mean=0.2, sd=0.3, low=0.1, upp=0.9,
                                                                     rng_engine=rng_engine)
    dummy_bbox = box_cxcywh2xyxy(dummy_bbox)
    dummy_bbox = np.clip(dummy_bbox, 0., 1.)
    dummy_bbox[0] *= image_size[0]
    dummy_bbox[1] *= image_size[1]
    dummy_bbox[2] *= image_size[0]
    dummy_bbox[3] *= image_size[1]

    return dummy_bbox
