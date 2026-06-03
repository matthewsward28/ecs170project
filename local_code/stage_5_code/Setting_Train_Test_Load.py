'''
Concrete Setting class for managing sequential train, test, save, and evaluation pipelines
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

class Setting_Train_Test_Load:
    def __init__(self, setting_name, setting_description):
        self.setting_name = setting_name
        self.setting_description = setting_description

    def prepare(self, data_obj, method_obj, result_obj, evaluate_obj):
        self.dataset = data_obj
        self.method = method_obj
        self.result = result_obj
        self.evaluate = evaluate_obj

    def load_run_save_evaluate(self):
        # load data and build vocabulary map matrices
        loaded_data = self.dataset.load()
        train_set = loaded_data['train']
        test_set = loaded_data['test']

        # Inject vocabulary configurations into the text architecture model components dynamically
        # Text models require explicit mapping definitions to scale their word embedding arrays safely
        if hasattr(self.dataset, 'vocab') and len(self.dataset.vocab) > 0:
            vocab_size = len(self.dataset.vocab)
            # Re-initialize or assign the internal architecture matching the structural dimension limits
            self.method.__init__(
                mName=self.method.method_name, 
                mDescription=self.method.method_description, 
                vocab_size=vocab_size
            )

        # training model and capture loss history for plotting sequential curves
        print('training model...')
        history = self.method.train_model(train_set['X'], train_set['y'])

        # testing model and gathering predictions across hidden recurrent elements
        print('testing model...')
        prediction_y = self.method.test(test_set['X'])

        # save results matching format rules
        self.result.data = {'true_y': test_set['y'], 'pred_y': prediction_y}
        self.result.save()

        # evaluate results via metrics computation modules
        # return metrics and history for the script curves generation processing
        return self.evaluate.evaluate(prediction_y, test_set['y']), history