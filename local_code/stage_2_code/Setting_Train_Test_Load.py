
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
        
        # Separate train and test sets
        train_set = loaded_data['train']
        test_set = loaded_data['test']

        # Training model
        self.method.train(train_set['X'], train_set['y'])

        # Testing model
        prediction_y = self.method.test(test_set['X'])

        # Save results
        self.result.data = {'true_y': test_set['y'], 'pred_y': prediction_y}
        self.result.save()

        # Evaluate results
        return self.evaluate.evaluate(prediction_y, test_set['y'])