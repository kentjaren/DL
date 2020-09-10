import logging

def CreateModel(opt):
    model = None
    if opt.model in ['ReconNet', 'DeepInverse']:
        from .ensemble_net import EnsembleNet
        model = EnsembleNet(opt)
    else:
        raise ValueError('Model {} was not recongnized.'.format(opt.model))

    logging.info('model {} was created.'.format(opt.model))
    model.init_weights()

    if opt.load_path:
        assert len(opt.load_path) == 3
        for idx, path in enumerate(opt.load_path):
            if path != 'placeholder':
                model.load_model(idx, path)

    return model
