from lib.test.utils import TrackerParams
from lib.test.evaluation.environment import env_settings
from lib.config.ostrack.config import cfg, update_config_from_file
from lib.test.parameter.ostrack_checkpoint_registry import (
    parse_parameter_name,
    resolve_paths,
    verify_checkpoint,
)


def parameters(parameter_name: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    spec = parse_parameter_name(
        parameter_name, expected_variant="e0", expected_version="v1")
    yaml_file, checkpoint_file = resolve_paths(spec, prj_dir, save_dir)
    verify_checkpoint(spec, checkpoint_file)
    update_config_from_file(str(yaml_file))
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    params.checkpoint = str(checkpoint_file)
    params.checkpoint_id = spec.checkpoint_id
    params.checkpoint_sha256 = spec.sha256
    params.config_name = spec.config_name

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
