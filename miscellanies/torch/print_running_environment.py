import torch


def print_running_environment(args):
    device = torch.device(args.device)
    from miscellanies.cpu_info import get_processor_name
    print(f'CPU: {get_processor_name()}')
    if 'cuda' in device.type:
        if torch.cuda.is_available():
            print(f'GPU: {torch.cuda.get_device_name(device)}')
