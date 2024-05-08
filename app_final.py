import os
import argparse
import threading

import main_app 
from tetris import Tetris

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device_no)
    p1 = threading.Thread(target=main_app.solve, args=(args,)) # thêm dấu (,) ở đuôi args vì nó yêu cầu đuôi là một iterable
    p1.start()
    
    p2 = threading.Thread(target=Tetris(16, 30).run, args=())
    p2.start()
    
    