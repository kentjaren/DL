from base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.is_Train = False
        self.parser.add_argument('--batchsize', type=int, default=16, help='input batch size')
