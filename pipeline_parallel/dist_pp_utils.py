from .dist_gpipe_pipeline_async import GpipeAsync
from modules.dist_deberta_pp_module import *


def get_pp_module(args, config, device, use_dp):
    if args.pp_mode == 'gpipe':
        return GpipeAsync(args, config, device, use_dp)
    else:
        print("Not recognize this pipeline parallel mode.")
        assert False
        
def get_deberta_pp_module(args, config, device, use_dp):
    if args.pp_mode == 'gpipe':
        return GpipeAsync(
            args, config, device, use_dp,
            _StageFirst=DebertaStageFirst,
            _StageLast=DebertaStageLast,
            _StageMiddle=DebertaStageMiddle,
        )
    else:
        print("Not recognize this pipeline parallel mode.")
        assert False
