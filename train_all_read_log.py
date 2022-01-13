import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
path_name = './re.log'
save_name = './val.jpg'
plot_details = True
K = 4 # Keep 4 decimals

epoch_list = []
best_score_list = []
val_score_list = []
patience_flag_list = []
learning_rate_0 = []
learning_rate_1 = []

for content in open(path_name, 'r', encoding='UTF-8'):
    _a = content.find('/400')
    _b = content.find('train_loss:')
    _c = content.find('min_loss:')
    _f = content.find('patience_flag:')
    _d = content.find('lr_coder:')
    _e = content.find('lr_transformer:')

    if _a >= 0:
        epoch = int(content.split('/400')[0].split('| ')[-1])
        if epoch != 0:
            epoch_list.append(epoch)
    if _c >= 0:
        best_score = float(content[_c:][:-1].split(' ')[-1])
        best_score_list.append(best_score)
    if _b >= 0:
        val_score = float(content[_b:][:-1].split(' ')[-1])
        val_score_list.append(val_score)
    if _f >= 0:
        patience_flag_list.append(int(content.split(' ')[2]))
    if _d >= 0:
        learning_rate_0.append(float(content.split(' ')[6]))
    if _e >= 0:
        learning_rate_1.append(float(content.split(' ')[10][:-1]))

print('No | train_L | min_L | f |       lr')

if epoch_list[-1] == epoch_list[-2]:
    # avoid duplicate items
    epoch_list.pop()

for i in range(len(epoch_list)):
    print ('%1.3d %8.4f %8.4f % 1.1d %9.2e %8.2e' % (epoch_list[i], 
        val_score_list[i], best_score_list[i], patience_flag_list[i],
        learning_rate_0[i], learning_rate_1[i]))

if plot_details:
    plt.plot(epoch_list, best_score_list, 'r*-', Markersize=1, label='min loss')
    plt.plot(epoch_list, val_score_list, 'go-', Markersize=1, label='train loss')
    plt.title('PSNR train score')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(save_name)
