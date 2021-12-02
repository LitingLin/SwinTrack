def numpy_no_multithreading():
    try:
        import mkl
        mkl.set_num_threads(1)
    except ImportError:
        pass
    import os
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
