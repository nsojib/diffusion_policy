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


def main(log_fn, topk):
    print(f"Loading log file: {log_fn}")
    logs = []
    with open(log_fn, "r") as f:
        for line in f:
            logs.append(json.loads(line))
    
    print(f"Number of logs: {len(logs)}")


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

    print(f"Number of epochs: {len(epoch_info)}") 
    print(f"Epochs: {sorted(epoch_info.keys())}")

    epochs = np.array( sorted(epoch_info.keys()) )
    train_loss =np.array(  [epoch_info[e]['train_loss'] for e in epochs] )
    val_loss   = np.array( [epoch_info[e]['val_loss']   for e in epochs] )
    train_score =np.array(  [epoch_info[e]['train/mean_score'] for e in epochs] )
    test_score  =np.array(  [epoch_info[e]['test/mean_score']  for e in epochs] )
    action_mse = np.array( [epoch_info[e]['train_action_mse_error'] for e in epochs] )

     
    idsx_val_min = np.argsort(val_loss)[:topk]
    idsx_test_max= np.argsort(test_score)[::-1][:topk] 



    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

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
    axs[0].set_ylim(-0.01, 0.25)

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

    # Final formatting
    plt.xticks(epochs, rotation=45)
    plt.tight_layout()


    base_path = os.path.dirname(log_fn)
    save_path = os.path.join(base_path, "log_view.png")
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



# python log2plot.py --log_fn /home/ns1254/diffusion_policy/data/dp_logs/coffee1/logs.json.txt 
# python log2plot.py --log_fn /home/ns1254/diffusion_policy/data/dp_logs/coffee2/logs.json.txt 
# python log2plot.py --log_fn /home/ns1254/diffusion_policy/data/dp_logs/coffee3/logs.json.txt

# python log2plot.py --log_fn /home/ns1254/diffusion_policy/data/dp_logs/kitchen1/logs.json.txt


