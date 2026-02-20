"""
Track on MOT15 dataset using ByteTrack
Adapted from tools/track.py for MOT15 specific evaluation
"""
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, setup_logger
from yolox.evaluators import MOTEvaluator

import argparse
import os
import random
import warnings


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack MOT15 Evaluation")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # Experiment file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file (e.g., exps/example/mot/yolox_x_mix_det.py)",
    )
    
    # Model checkpoint
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint for eval")
    
    # Device settings
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    
    # Batch size
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
    
    # Model settings
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    
    # Test settings
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test set.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val_half",
        help="split to evaluate (train, val_half, train_half, test)",
    )
    
    # Detection parameters
    parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    
    # Tracking parameters
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    
    return parser


@logger.catch
def main(exp, args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting."
        )

    cudnn.benchmark = True

    # Setup output directory
    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    results_folder = os.path.join(file_name, "track_results")
    os.makedirs(results_folder, exist_ok=True)

    setup_logger(file_name, distributed_rank=0, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    # Update experiment settings
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    # Load model
    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    # Setup evaluator
    val_loader = exp.get_eval_loader(args.batch_size, is_distributed=False, testdev=args.test)
    evaluator = MOTEvaluator(
        args=args,
        dataloader=val_loader,
        img_size=exp.test_size,
        confthre=exp.test_conf,
        nmsthre=exp.nmsthre,
        num_classes=exp.num_classes,
    )

    # Setup device
    if args.device == "gpu":
        model.cuda()
    model.eval()

    # Load checkpoint
    if args.ckpt is None:
        ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
    else:
        ckpt_file = args.ckpt
    
    logger.info(f"Loading checkpoint from {ckpt_file}")
    
    if args.device == "gpu":
        loc = "cuda:0"
    else:
        loc = "cpu"
    
    ckpt = torch.load(ckpt_file, map_location=loc)
    model.load_state_dict(ckpt["model"])
    logger.info("Checkpoint loaded successfully.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()

    # Run evaluation
    logger.info("Starting tracking on MOT15...")
    *_, summary = evaluator.evaluate(
        model,
        distributed=False,
        half=args.fp16,
        result_folder=results_folder
    )
    logger.info("\n" + summary)


if __name__ == "__main__":
    args = make_parser().parse_args()
    
    # Set default experiment file if not provided
    if args.exp_file is None:
        args.exp_file = "exps/example/mot/yolox_x_mix_det.py"
        logger.info(f"Using default experiment file: {args.exp_file}")
    
    # Set default checkpoint if not provided
    if args.ckpt is None:
        args.ckpt = "pretrained/bytetrack_x_mot17.pth.tar"
        logger.info(f"Using default checkpoint: {args.ckpt}")
    
    # Set experiment name
    if args.experiment_name is None:
        args.experiment_name = "mot15_{}".format(args.split if not args.test else "test")
    
    # Load experiment
    exp = get_exp(args.exp_file, args.name)
    
    # Override data path to use MOT15
    exp.data_dir = "datasets/mot15"
    exp.train_ann = "train.json"
    exp.val_ann = "{}.json".format(args.split)
    exp.test_ann = "test.json" if args.test else "{}.json".format(args.split)
    
    logger.info(f"Dataset: MOT15")
    logger.info(f"Data directory: {exp.data_dir}")
    logger.info(f"Annotation file: {exp.test_ann if args.test else exp.val_ann}")
    
    main(exp, args)
