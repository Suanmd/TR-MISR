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
learning_rate_0_dir = {}
learning_rate_1_dir = {}

for content in open(path_name, 'r', encoding='UTF-8'):
    _b = content.find('best_score:')
    _c = content.find('val_score:')
    _d = content.find('learning rate of group 0')
    _e = content.find('learning rate of group 1')
    if _b >= 0:
        best_score = float(content[_b:][:-1].split(' ')[-1])
        best_score_list.append(best_score)
    if _c >= 0:
        val_score = float(content[_c:][:-1].split(' ')[-1])
        val_score_list.append(val_score)
    if _d >= 0:
        learning_rate_0_dir[int(content.split(':')[0].split(' ')[-1])] = content[_d+28:-1]
    if _e >= 0:
        learning_rate_1_dir[int(content.split(':')[0].split(' ')[-1])] = content[_e+28:-1]

print('epoch  |  val_score  |  best_score')

for i in range(len(val_score_list)):
    print ('%4.3d %14.5f %13.5f'%(i+1, val_score_list[i], best_score_list[i]))
    if (i+1) in learning_rate_0_dir.keys():
        print('Epoch %d reducing lr of group 0: %s' % (i+1, learning_rate_0_dir[i+1]))
    if (i+1) in learning_rate_1_dir.keys():
        print('Epoch %d reducing lr of group 1: %s' % (i+1, learning_rate_1_dir[i+1]))

epoch_list = [(i+1) for i in range(len(val_score_list))]

if plot_details:
    plt.plot(epoch_list, best_score_list, 'r*-', Markersize=1, label='best score')
    plt.plot(epoch_list, val_score_list, 'go-', Markersize=1, label='val score')
    plt.title('PSNR val score')
    plt.xlabel('epoch')
    plt.ylabel('cPSNR')
    plt.legend()
    plt.savefig(save_name)