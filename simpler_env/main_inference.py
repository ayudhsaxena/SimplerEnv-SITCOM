import os

import numpy as np
import tensorflow as tf

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator

try:
    from simpler_env.policies.sitcom.sitcom import SITCOMInference
except ImportError as e:
    print("SITCOM is not correctly imported.")
    print(e)

try:
    from simpler_env.policies.octo.octo_model import OctoInference
except ImportError as e:
    print("Octo is not correctly imported.")
    print(e)


if __name__ == "__main__":
    args = get_args()
    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        # prevent a single tf process from taking up all the GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
        )
    print(f"**** {args.policy_model} ****")
    # policy model creation; update this if you are using a new policy model
    if args.policy_model == "rt1":
        from simpler_env.policies.rt1.rt1_model import RT1Inference
        assert args.ckpt_path is not None
        model = RT1Inference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )
    elif "octo" in args.policy_model:
        from simpler_env.policies.octo.octo_server_model import OctoServerInference
        if args.ckpt_path is None or args.ckpt_path == "None":
            args.ckpt_path = args.policy_model
        if "server" in args.policy_model:
            model = OctoServerInference(
                model_type=args.ckpt_path,
                policy_setup=args.policy_setup,
                action_scale=args.action_scale,
            )
        else:
            model = OctoInference(
                model_type=args.ckpt_path,
                policy_setup=args.policy_setup,
                init_rng=args.octo_init_rng,
                action_scale=args.action_scale,
            )
    elif args.policy_model == "openvla":
        assert args.ckpt_path is not None
        from simpler_env.policies.openvla.openvla_model import OpenVLAInference
        model = OpenVLAInference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )
    elif args.policy_model == "cogact":
        from simpler_env.policies.sim_cogact import CogACTInference
        assert args.ckpt_path is not None
        model = CogACTInference(
            saved_model_path=args.ckpt_path,  # e.g., CogACT/CogACT-Base
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
            action_model_type='DiT-L',
            cfg_scale=1.5                     # cfg from 1.5 to 7 also performs well
        )
    elif args.policy_model == "spatialvla":
        assert args.ckpt_path is not None
        from simpler_env.policies.spatialvla.spatialvla_model import SpatialVLAInference
        model = SpatialVLAInference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )
    elif args.policy_model == "sitcom":
        assert args.ckpt_path is not None
        from simpler_env.policies.sitcom.sitcom import SITCOMInference
        model = SITCOMInference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
            # Add the new planning parameters here
            num_initial_actions=args.num_initial_actions if hasattr(args, 'num_initial_actions') else 10,
            horizon_per_action=args.horizon_per_action if hasattr(args, 'horizon_per_action') else 5,
            num_steps_ahead=args.num_steps_ahead if hasattr(args, 'num_steps_ahead') else 3,
            num_candidates=args.num_candidates if hasattr(args, 'num_candidates') else 5,
            num_best_actions=args.num_best_actions if hasattr(args, 'num_best_actions') else 3,
            temperature=args.temperature if hasattr(args, 'temperature') else 1.0,
            render_tree=args.render_tree if hasattr(args, 'render_tree') else False,
            logging_dir=args.logging_dir,
        )
    else:
        raise NotImplementedError()

    # run real-to-sim evaluation
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
