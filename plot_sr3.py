import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py




def read_3seed_sr(fn_3seed):
    with open(fn_3seed, "r") as f:
        lines=f.readlines()
    sr=eval(lines[4].strip().split(",")[1].strip())
    return sr



def main(checkpoint_dir):
    
    files=os.listdir(checkpoint_dir) 
    files=[f for f in files if os.path.isdir(os.path.join(checkpoint_dir, f)) and f.startswith("epoch=")]


    sr_info={}
    for filename in files:
        fn_3seed=os.path.join(checkpoint_dir, filename, "eval_scores.txt")
        sr=read_3seed_sr(fn_3seed)
        training_time_sr = eval( filename.split("=")[-1].strip() )
        epoch = int(filename.split("=")[1].split("-")[0].strip())
        # print(f"Epoch: {epoch}, training time SR: {training_time_sr}, 3 seed SR: {sr}")
        sr_info[epoch] = {
            'training_time_sr': training_time_sr,
            '3seed_sr': sr
        }


    epochs = np.array( sorted(sr_info.keys()) )
    training_time_sr =np.array(  [sr_info[e]['training_time_sr'] for e in epochs] )
    test_time_sr = np.array( [sr_info[e]['3seed_sr'] for e in epochs] )
    print(f"Number of epochs: {len(sr_info)}")
    print(f"Epochs: {sorted(sr_info.keys())}")
    print(f"Training time SR: {training_time_sr}")
    print(f"Test time SR: {test_time_sr}")

 
    save_fn=os.path.join(os.path.dirname(checkpoint_dir), "sr_info.png")

    # plot epoch vs training time SR and test time SR
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, training_time_sr, label='Training time SR', color='blue')
    plt.plot(epochs, test_time_sr, label='Test time SR', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('SR')
    plt.title('Epoch vs SR')
    plt.xticks(epochs, rotation=45)
    plt.ylim(0.6, 1)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_fn)
    print(f"Saved SR plot to {save_fn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate plot from dp log data.')
    parser.add_argument('--checkpoint_dir', type=str, help='checkpoint_dir name') 
    args = parser.parse_args() 
    main(args.checkpoint_dir)



# python plot_sr3.py --checkpoint_dir /home/ns1254/diffusion_policy/data/outputs/can_mh_img1/checkpoints
# python plot_sr3.py --checkpoint_dir /home/ns1254/diffusion_policy/data/dp_logs/can_bc_mh_img_better/checkpoints



