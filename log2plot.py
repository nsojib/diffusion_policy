# %%
import numpy as np
import matplotlib.pyplot as plt
import json 
import os 
import argparse

def highlight_extrema(ax, x, y, color, marker, label):
    x = np.array(x)
    y = np.array(y)
    y_max, y_min = y.max(), y.min()
    idxs_max = np.where(y == y_max)[0]
    idxs_min = np.where(y == y_min)[0]
    ax.scatter(x[idxs_max], y[idxs_max],
               color=color, s=100, marker='*', edgecolor='k')
    ax.scatter(x[idxs_min], y[idxs_min],
               color=color, s=100, marker='X', edgecolor='k')


def annotate_points(ax, x, y, idxs, va='top'):
    
    yt=-15 if va=='bottom' else 15
    
    for i in idxs:
        ax.annotate(f'{y[i]:.3f}',
                    xy=(x[i], y[i]),
                    xytext=(0, yt),
                    textcoords='offset points',
                    ha='center',
                    va=va,
                    fontsize='small',
                    fontweight='bold')  

def read_3seed_sr(fn_3seed):
    with open(fn_3seed, "r") as f:
        lines=f.readlines()
    sr=eval(lines[4].strip().split(",")[1].strip())
    return sr


def read_3seed_sr_from_log(checkpoint_dir):

    files=os.listdir(checkpoint_dir) 
    files=[f for f in files if os.path.isdir(os.path.join(checkpoint_dir, f)) and f.startswith("epoch=")]

    logdir = str(os.path.dirname(checkpoint_dir)).split('/')[-1]
    print(f"Log directory: {logdir}")

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
    print('Epochs sr3:', epochs)
    return sr_info
    # epochs = np.array( sorted(sr_info.keys()) )
    # training_time_sr =np.array(  [sr_info[e]['training_time_sr'] for e in epochs] )
    # test_time_sr = np.array( [sr_info[e]['3seed_sr'] for e in epochs] )
    # print(f"Number of epochs: {len(sr_info)}")
    # print(f"Epochs: {sorted(sr_info.keys())}")
    # print(f"Training time SR: {training_time_sr}")
    # print(f"Test time SR: {test_time_sr}")
    # return epochs, training_time_sr, test_time_sr 


def main(log_fn, topk):
    print(f"Loading log file: {log_fn}")
    logs = []
    with open(log_fn, "r") as f:
        for line in f:
            logs.append(json.loads(line))
    
    print(f"Number of logs: {len(logs)}")

    logdir = str(os.path.dirname(log_fn)).split('/')[-1]
    print(f"Log directory: {logdir}")
    # return 

    checkpoint_dir = os.path.join(os.path.dirname(log_fn), "checkpoints")
    sr_info= read_3seed_sr_from_log(checkpoint_dir)


    epoch_info={}
    for log in logs:
        if len(log)>6:
            epoch=log['epoch']
            epoch_info[epoch] = {
                'train_loss': log['train_loss'],
                'val_loss': log['val_loss'],
                'train/mean_score': log['train/mean_score'],
                'test/mean_score': log['test/mean_score'],
                'train_action_mse_error': log['train_action_mse_error']
            }
            epoch_info[epoch]['training_time_sr'] = sr_info[epoch]['training_time_sr']
            epoch_info[epoch]['test_time_sr'] = sr_info[epoch]['3seed_sr']

    print(f"Number of epochs: {len(epoch_info)}") 
    print(f"Epochs: {sorted(epoch_info.keys())}")

    epochs = np.array( sorted(epoch_info.keys()) )
    train_loss =np.array(  [epoch_info[e]['train_loss'] for e in epochs] )
    val_loss   = np.array( [epoch_info[e]['val_loss']   for e in epochs] )
    train_score =np.array(  [epoch_info[e]['train/mean_score'] for e in epochs] )
    test_score  =np.array(  [epoch_info[e]['test/mean_score']  for e in epochs] )
    action_mse = np.array( [epoch_info[e]['train_action_mse_error'] for e in epochs] )

    training_time_sr3 = np.array( [epoch_info[e]['training_time_sr'] for e in epochs] )
    test_time_sr3 = np.array( [epoch_info[e]['test_time_sr'] for e in epochs] )

     
    idsx_val_min = np.argsort(val_loss)[:topk]
    idsx_test_max= np.argsort(test_score)[::-1][:topk] 



    fig, axs = plt.subplots(4, 1, figsize=(10, 6), sharex=True)

    # 1. Loss vs Epoch
    axs[0].plot(epochs, train_loss, label='Train Loss', linestyle='-')
    axs[0].plot(epochs, val_loss,   label='Val   Loss',   linestyle='--')
    axs[0].scatter(epochs, train_loss, color='tab:blue',  s=10, alpha=0.5)
    axs[0].scatter(epochs, val_loss,   color='tab:orange',s=10, alpha=0.5)

    highlight_extrema(axs[0], epochs, train_loss, 'tab:blue',  '*', 'Train')
    highlight_extrema(axs[0], epochs, val_loss,   'tab:orange','X', 'Val')


    axs[0].scatter(epochs[idsx_val_min], val_loss[idsx_val_min],
                color='tab:orange', s=100, marker='v', edgecolor='k')
    annotate_points(axs[0], epochs, val_loss, idsx_val_min, va='top')

    annotate_points(axs[1], epochs, test_score, idsx_val_min, va='top')  #corresponding test_score for the top val_loss

    # --------------------------------------

    axs[0].set_ylabel('Loss')
    axs[0].legend(loc='upper right', fontsize='small', ncol=2)
    axs[0].grid(alpha=0.7)
    # axs[0].set_ylim(-0.01, 0.25)

    # 2. Mean Score vs Epoch
    axs[1].plot(epochs, train_score, label='Train Mean Score', linestyle='-')
    axs[1].plot(epochs, test_score,  label='Test  Mean Score',  linestyle='--')
    axs[1].scatter(epochs, train_score, color='tab:blue',  s=10, alpha=0.5)
    axs[1].scatter(epochs, test_score,  color='tab:orange',s=10, alpha=0.5)

    highlight_extrema(axs[1], epochs, train_score, 'tab:blue',  '*', 'Train')
    highlight_extrema(axs[1], epochs, test_score,  'tab:orange','X', 'Test')


    axs[1].scatter(epochs[idsx_test_max], test_score[idsx_test_max],
                color='tab:green', s=100, marker='^', edgecolor='k')
    annotate_points(axs[1], epochs, test_score, idsx_test_max, va='bottom')


    annotate_points(axs[0], epochs, val_loss, idsx_test_max, va='top')  #corresponding val_loss for the top test scores



    axs[1].set_ylabel('Mean Score')
    axs[1].legend(loc='lower right', fontsize='small', ncol=2)
    axs[1].grid(alpha=0.7)
    axs[1].set_ylim(0.3, 1.1)

    # 3. Action MSE Error vs Epoch
    axs[2].plot(epochs, action_mse, label='Train Action MSE Error', linestyle='-')
    axs[2].scatter(epochs, action_mse, color='tab:red', s=10, alpha=0.5)

    highlight_extrema(axs[2], epochs, action_mse, 'tab:red', 'X', 'MSE')

    # annotate mse error for both the top val_loss and top test_score 
    annotate_points(axs[2], epochs, action_mse, idsx_val_min, va='top')
    annotate_points(axs[2], epochs, action_mse, idsx_test_max, va='top')


    axs[2].set_ylabel('MSE Error')
    axs[2].set_xlabel('Epoch')
    axs[2].legend(loc='upper right', fontsize='small')
    axs[2].grid(alpha=0.7)
    axs[2].set_ylim(-0.01, 0.04)


    idsx_test3_max= np.argsort(test_time_sr3)[::-1][:topk] 
    # 4. SR vs Epoch
    axs[3].plot(epochs, training_time_sr3, label='Training Time SR', linestyle='-')
    axs[3].plot(epochs, test_time_sr3, label='Test Time SR', linestyle='--')
    axs[3].scatter(epochs, training_time_sr3, color='tab:blue',  s=10, alpha=0.5)
    axs[3].scatter(epochs, test_time_sr3,  color='tab:orange',s=10, alpha=0.5)
    # highlight_extrema(axs[3], epochs, training_time_sr3, 'tab:blue',  '*', 'Train')
    highlight_extrema(axs[3], epochs, test_time_sr3,  'tab:orange','X', 'Test')
    # axs[3].scatter(epochs[idsx_val_min], training_time_sr3[idsx_val_min],
    #             color='tab:blue', s=100, marker='v', edgecolor='k')
    axs[3].scatter(epochs[idsx_test3_max], test_time_sr3[idsx_test3_max],
                color='tab:orange', s=100, marker='^', edgecolor='k')
    # annotate_points(axs[3], epochs, training_time_sr3, idsx_val_min, va='top')


    
    annotate_points(axs[3], epochs, test_time_sr3, idsx_test3_max, va='top')


    # axs[3].scatter(epochs[idsx_test_max], test_score[idsx_test_max],
    #             color='tab:green', s=100, marker='^', edgecolor='k') 



    axs[3].set_ylabel('SR3')
    axs[3].set_xlabel('Epoch')
    axs[3].legend(loc='lower right', fontsize='small', ncol=2)
    axs[3].grid(alpha=0.7)

    # Final formatting
    plt.xticks(epochs, rotation=45)
    plt.tight_layout()


    base_path = os.path.dirname(log_fn)
    save_path = os.path.join(base_path, f"{logdir}_log_view_sr3.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.close()

    plt.show()

    print(f"Saved to {save_path}") 



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate plot from dp log data.')
    parser.add_argument('--log_fn', type=str, help='log file name')
    parser.add_argument('--topk', type=int, default=3, help='number of top k points to highlight')
    args = parser.parse_args() 
    main(args.log_fn, args.topk)



# python log2plot.py --log_fn /home/carl_lab/ns/diffusion_policy/data/outputs/can_img_bc_b/logs.json.txt 
# python log2plot.py --log_fn /home/carl_lab/ns/diffusion_policy/data/outputs/can_img_bc_w/logs.json.txt 
# python log2plot.py --log_fn /home/carl_lab/ns/diffusion_policy/data/outputs/can_img_bc_wb/logs.json.txt

# python log2plot.py --log_fn /home/carl_lab/ns/diffusion_policy/data/outputs/lift_img_bc_b/logs.json.txt 
# python log2plot.py --log_fn /home/carl_lab/ns/diffusion_policy/data/outputs/lift_img_bc_w/logs.json.txt 
# python log2plot.py --log_fn /home/carl_lab/ns/diffusion_policy/data/outputs/lift_img_bc_wb/logs.json.txt


