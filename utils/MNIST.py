import torch


def eval_on_real_data(CGAN, num_epochs, es):
    CGAN.init_evaluator(CGAN.train_gen, CGAN.val_gen)
    CGAN.netE.train_evaluator(num_epochs=num_epochs, eval_freq=1, es=es)
    _, og_result = CGAN.netE.eval_once(CGAN.test_gen)
    return og_result


def convert_y_to_one_hot(y):
    new_y = torch.zeros([len(y), 10], dtype=torch.uint8, device='cpu')
    y = y.view(-1, 1)
    new_y.scatter_(1, y, 1)
    return new_y
