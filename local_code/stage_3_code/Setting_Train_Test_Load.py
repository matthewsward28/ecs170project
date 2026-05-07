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
        # load data
        loaded_data = self.dataset.load()
        train_set = loaded_data['train']
        test_set = loaded_data['test']

        # training model and capture loss history for plotting
        print('training model...')
        history = self.method.train(train_set['X'], train_set['y'])

        # testing model
        print('testing model...')
        prediction_y = self.method.test(test_set['X'])

        # save results
        self.result.data = {'true_y': test_set['y'], 'pred_y': prediction_y}
        self.result.save()

        # evaluate results
        # return metrics and history for the script to use
        return self.evaluate.evaluate(prediction_y, test_set['y']), history