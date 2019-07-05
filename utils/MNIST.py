def eval_on_real_data(CGAN, num_epochs, es):
    CGAN.init_evaluator(CGAN.train_gen, CGAN.val_gen)
    CGAN.netE.train_evaluator(num_epochs=num_epochs, eval_freq=1, es=es)
    _, og_result = CGAN.netE.eval_once(CGAN.test_gen)
    return og_result
