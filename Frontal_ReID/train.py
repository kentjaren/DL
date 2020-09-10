import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from model.model_Loader import CreateModel
from util.utils import error

opt = TrainOptions().parse()

data_loader = CreateDataLoader(opt)
model = CreateModel(opt)

err = error(model.save_dir)
for epoch in range(opt.count_epoch+1,  opt.epochs+1):
    epoch_start_time = time.time()
    err.initialize()

    for i, data in enumerate(data_loader):
        model.forward(data)

        model.optimize_G_parameters()
        if(i % opt.D_interval == 0):
            model.optimize_D_parameters()

        err.add(model.Loss_G.data[0], model.Loss_D.data[0])

    err.print_errors(epoch)
    print('End of epoch {0} \t Time Taken: {1} sec\n'.format(epoch, time.time()-epoch_start_time))
    model.save_result(epoch)
    if epoch % opt.save_epoch_freq == 0:
        print('Saving the model at the end of epoch {}\n'.format(epoch))
        model.save(epoch)
