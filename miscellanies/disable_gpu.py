def disable_gpu():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
