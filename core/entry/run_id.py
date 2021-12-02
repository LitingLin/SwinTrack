from datetime import datetime
import os


def generate_run_id(args):
    parts = [args.method_name, args.config_name]
    if args.mixin_config is not None:
        for mixin_config in args.mixin_config:
            parts.append('mixin')
            parts.append(os.path.splitext(os.path.basename(mixin_config))[0])
    parts.append(datetime.now().strftime("%Y.%m.%d-%H.%M.%S-%f"))

    return '-'.join(parts)
