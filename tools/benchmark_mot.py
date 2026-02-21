import argparse
import os
import os.path as osp
import time
import cv2
import torch
import pandas as pd
import glob
from loguru import logger
from collections import OrderedDict
from pathlib import Path

import motmetrics as mm
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack MOT Benchmark!")
    parser.add_argument("--mot_root", type=str, required=True, help="path to MOT15/16/17/20 root folder")
    parser.add_argument("-f", "--exp_file", type=str, default=None, help="pls input your expriment description file")
    parser.add_argument("-c", "--ckpt", type=str, default=None, help="ckpt for eval")
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fp16", action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--csv_path", type=str, default="evaluation_results.csv", help="path to save the results csv")
    
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", action="store_true", help="test mot20.")
    return parser

class Predictor(object):
    def __init__(self, model, exp, device=torch.device("cpu"), fp16=False):
        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img = cv2.imread(img)
        
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()

        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs, img_info

def run_sequence(predictor, seq_path, args):
    img_dir = osp.join(seq_path, "img1")
    if not osp.exists(img_dir):
        logger.warning(f"Image directory {img_dir} not found. Skipping.")
        return None
    
    files = []
    for ext in IMAGE_EXT:
        files.extend(glob.glob(osp.join(img_dir, "*" + ext)))
    files.sort()
    
    if not files:
        logger.warning(f"No images found in {img_dir}. Skipping.")
        return None

    tracker = BYTETracker(args, frame_rate=30)
    results = []

    logger.info(f"Processing sequence: {osp.basename(seq_path)}")
    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], predictor.test_size)
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    # result line: <frame>, <id>, <x>, <y>, <w>, <h>, <conf>, -1, -1, -1
                    results.append([frame_id, tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3], t.score, -1, -1, -1])
        
        if frame_id % 100 == 0:
            logger.info(f"  Frame {frame_id}/{len(files)}")
            
    return results

def evaluate(gt_path, res_data, seq_name):
    # Load ground truth
    gt = mm.io.loadtxt(gt_path, fmt='mot15-2D', min_confidence=1)
    
    # Load results from memory (formatted as list of lists)
    # We convert results to a dataframe compatible with motmetrics
    res_df = pd.DataFrame(res_data, columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility', 'Unused'])
    res_df = res_df.set_index(['FrameId', 'Id'])
    
    acc = mm.utils.compare_to_groundtruth(gt, res_df, 'iou', distth=0.5)
    return acc

def main(exp, args):
    output_dir = osp.join(exp.output_dir, args.name if args.name else exp.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    model.eval()

    if args.ckpt:
        logger.info(f"Loading checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
    
    if args.fuse:
        model = fuse_model(model)
    if args.fp16:
        model = model.half()

    predictor = Predictor(model, exp, args.device, args.fp16)
    
    seqs = [d for d in os.listdir(args.mot_root) if osp.isdir(osp.join(args.mot_root, d))]
    seqs.sort()
    
    all_accs = []
    all_names = []
    
    for seq in seqs:
        seq_path = osp.join(args.mot_root, seq)
        gt_path = osp.join(seq_path, "gt", "gt.txt")
        if not osp.exists(gt_path):
            logger.warning(f"Ground truth not found for {seq} at {gt_path}. Skipping evaluation.")
            continue
            
        res_data = run_sequence(predictor, seq_path, args)
        if res_data:
            acc = evaluate(gt_path, res_data, seq)
            all_accs.append(acc)
            all_names.append(seq)

    if not all_accs:
        logger.error("No sequences were evaluated.")
        return

    logger.info("Computing metrics...")
    mh = mm.metrics.create()
    metrics = mm.metrics.motchallenge_metrics
    summary = mh.compute_many(all_accs, names=all_names, metrics=metrics, generate_overall=True)
    
    # Format and save to CSV
    # We want MOTA, MOTP, IDF1, Precision, Recall specifically as requested
    # The summary already contains these.
    
    # Rename columns for clarity if needed, or just select them
    selected_metrics = ['mota', 'motp', 'idf1', 'precision', 'recall']
    # Check if they exist in summary (they should)
    printable_summary = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
    print(printable_summary)
    
    # Save to CSV
    # summary is a pandas DataFrame
    summary['model_name'] = args.name if args.name else exp.exp_name
    summary['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # If csv exists, append; otherwise write new
    if osp.exists(args.csv_path):
        summary.to_csv(args.csv_path, mode='a', header=False)
    else:
        summary.to_csv(args.csv_path)
    
    logger.info(f"Results saved to {args.csv_path}")

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
