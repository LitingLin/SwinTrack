def bbox_scale_and_translate(bbox, scale, input_center, output_center):
    '''
        (i - input_center) * scale = o - output_center
        :return XYXY format
    '''
    x1, y1, x2, y2 = bbox
    ic_x, ic_y = input_center
    oc_x, oc_y = output_center
    s_x, s_y = scale
    o_x1 = oc_x + (x1 - ic_x) * s_x
    o_y1 = oc_y + (y1 - ic_y) * s_y
    o_x2 = oc_x + (x2 - ic_x) * s_x
    o_y2 = oc_y + (y2 - ic_y) * s_y
    return (o_x1, o_y1, o_x2, o_y2)
