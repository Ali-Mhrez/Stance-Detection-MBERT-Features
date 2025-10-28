import os
import csv

def write_results(filename, run_id, eval_loss, eval_acc, eval_f1, eval_mf1, test_loss, test_acc, test_f1, test_mf1):
    if os.path.isfile(filename):
        file = open(filename, 'a', encoding='utf-8')
    else:
        file = open(filename, 'w', encoding='utf-8')
        writer = csv.writer(file)
        writer.writerow(['run_id', \
                     'eval_loss', 'eval_accuracy', 'eval_agree', 'eval_disagree', 'eval_discuss', 'eval_unrelated', 'eval_mf1', \
                     'test_loss', 'test_accuracy', 'test_agree', 'test_disagree', 'test_discuss', 'test_unrelated', 'test_mf1'])

    writer = csv.writer(file)
    writer.writerow([run_id, \
        eval_loss, eval_acc, eval_f1[0], eval_f1[1], eval_f1[2], eval_f1[3], eval_mf1, \
        test_loss, test_acc, test_f1[0], test_f1[1], test_f1[2], test_f1[3], test_mf1])
    file.close()
    
def set_seed(seed):
    import torch
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")