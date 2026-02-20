"""
Convert MOT15 dataset to COCO format for ByteTrack
Based on convert_mot17_to_coco.py but adapted for MOT15 format
"""
import os
import numpy as np
import json
import cv2


# MOT15 specific settings
DATA_PATH = 'datasets/mot15'
OUT_PATH = os.path.join(DATA_PATH, 'annotations')
SPLITS = ['train_half', 'val_half', 'train', 'test']
HALF_VIDEO = True
CREATE_SPLITTED_ANN = True
CREATE_SPLITTED_DET = True


if __name__ == '__main__':
    
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
    
    for split in SPLITS:
        if split == "test":
            data_path = os.path.join(DATA_PATH, 'test')
        else:
            data_path = os.path.join(DATA_PATH, 'train')
        
        out_path = os.path.join(OUT_PATH, '{}.json'.format(split))
        out = {'images': [], 'annotations': [], 'videos': [],
               'categories': [{'id': 1, 'name': 'pedestrian'}]}
        
        seqs = os.listdir(data_path)
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        tid_curr = 0
        tid_last = -1
        
        for seq in sorted(seqs):
            if '.DS_Store' in seq or seq.startswith('.'):
                continue
            
            video_cnt += 1
            out['videos'].append({'id': video_cnt, 'file_name': seq})
            seq_path = os.path.join(data_path, seq)
            img_path = os.path.join(seq_path, 'img1')
            ann_path = os.path.join(seq_path, 'gt/gt.txt')
            
            if not os.path.exists(img_path):
                print(f"Warning: {img_path} does not exist, skipping {seq}")
                continue
            
            images = os.listdir(img_path)
            num_images = len([image for image in images if 'jpg' in image or 'png' in image])
            
            if num_images == 0:
                print(f"Warning: No images found in {img_path}, skipping {seq}")
                continue
            
            if HALF_VIDEO and ('half' in split):
                image_range = [0, num_images // 2] if 'train' in split else \
                              [num_images // 2 + 1, num_images - 1]
            else:
                image_range = [0, num_images - 1]
            
            # Process images
            for i in range(num_images):
                if i < image_range[0] or i > image_range[1]:
                    continue
                
                # Try both .jpg and .png extensions
                img_file = os.path.join(data_path, '{}/img1/{:06d}.jpg'.format(seq, i + 1))
                if not os.path.exists(img_file):
                    img_file = os.path.join(data_path, '{}/img1/{:06d}.png'.format(seq, i + 1))
                
                if not os.path.exists(img_file):
                    print(f"Warning: Image {img_file} not found")
                    continue
                
                img = cv2.imread(img_file)
                if img is None:
                    print(f"Warning: Could not read {img_file}")
                    continue
                
                height, width = img.shape[:2]
                
                # Determine file extension
                ext = '.jpg' if img_file.endswith('.jpg') else '.png'
                
                image_info = {
                    'file_name': '{}/img1/{:06d}{}'.format(seq, i + 1, ext),
                    'id': image_cnt + i + 1,
                    'frame_id': i + 1 - image_range[0],
                    'prev_image_id': image_cnt + i if i > 0 else -1,
                    'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                    'video_id': video_cnt,
                    'height': height,
                    'width': width
                }
                out['images'].append(image_info)
            
            print('{}: {} images'.format(seq, num_images))
            
            # Process annotations (only for training data)
            if split != 'test':
                if not os.path.exists(ann_path):
                    print(f"Warning: Annotation file {ann_path} not found")
                    image_cnt += num_images
                    continue
                
                det_path = os.path.join(seq_path, 'det/det.txt')
                
                try:
                    anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
                    if anns.ndim == 1:
                        anns = anns.reshape(1, -1)
                except Exception as e:
                    print(f"Warning: Could not load annotations from {ann_path}: {e}")
                    image_cnt += num_images
                    continue
                
                # Load detections if available
                if os.path.exists(det_path):
                    try:
                        dets = np.loadtxt(det_path, dtype=np.float32, delimiter=',')
                        if dets.ndim == 1:
                            dets = dets.reshape(1, -1)
                    except Exception as e:
                        print(f"Warning: Could not load detections from {det_path}: {e}")
                        dets = None
                else:
                    dets = None
                
                # Create split annotations
                if CREATE_SPLITTED_ANN and ('half' in split):
                    anns_out = np.array([anns[i] for i in range(anns.shape[0])
                                        if int(anns[i][0]) - 1 >= image_range[0] and
                                        int(anns[i][0]) - 1 <= image_range[1]], np.float32)
                    if len(anns_out) > 0:
                        anns_out[:, 0] -= image_range[0]
                        gt_out = os.path.join(seq_path, 'gt/gt_{}.txt'.format(split))
                        fout = open(gt_out, 'w')
                        for o in anns_out:
                            fout.write('{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
                                        int(o[0]), int(o[1]), int(o[2]), int(o[3]), int(o[4]), int(o[5]),
                                        int(o[6]), int(o[7]), o[8]))
                        fout.close()
                
                # Create split detections
                if CREATE_SPLITTED_DET and ('half' in split) and dets is not None:
                    dets_out = np.array([dets[i] for i in range(dets.shape[0])
                                        if int(dets[i][0]) - 1 >= image_range[0] and
                                        int(dets[i][0]) - 1 <= image_range[1]], np.float32)
                    if len(dets_out) > 0:
                        dets_out[:, 0] -= image_range[0]
                        det_out = os.path.join(seq_path, 'det/det_{}.txt'.format(split))
                        dout = open(det_out, 'w')
                        for o in dets_out:
                            dout.write('{:d},{:d},{:.1f},{:.1f},{:.1f},{:.1f},{:.6f}\n'.format(
                                        int(o[0]), int(o[1]), float(o[2]), float(o[3]), float(o[4]), float(o[5]),
                                        float(o[6])))
                        dout.close()
                
                print('{} ann images'.format(int(anns[:, 0].max())))
                
                # Process annotations
                for i in range(anns.shape[0]):
                    frame_id = int(anns[i][0])
                    if frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]:
                        continue
                    
                    track_id = int(anns[i][1])
                    cat_id = int(anns[i][7])
                    ann_cnt += 1
                    
                    # MOT15 specific: all annotations are pedestrians (category_id = 1)
                    # MOT15 uses simpler annotation format compared to MOT17
                    category_id = 1
                    if track_id != tid_last:
                        tid_curr += 1
                        tid_last = track_id
                    
                    ann = {
                        'id': ann_cnt,
                        'category_id': category_id,
                        'image_id': image_cnt + frame_id,
                        'track_id': tid_curr,
                        'bbox': anns[i][2:6].tolist(),
                        'conf': float(anns[i][6]),
                        'iscrowd': 0,
                        'area': float(anns[i][4] * anns[i][5])
                    }
                    out['annotations'].append(ann)
            
            image_cnt += num_images
            print(f"Track IDs: current={tid_curr}, last={tid_last}")
        
        print('Loaded {} for {} images and {} samples'.format(
            split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))
        print(f"Saved to {out_path}\n")
