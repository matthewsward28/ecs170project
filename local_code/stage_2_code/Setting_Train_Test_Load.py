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
        # Load train and test sets
        loaded_train_data = self.dataset.load_train() 
        loaded_test_data = self.dataset.load_test()
        # Train model
        self.method.data = loaded_train_data
        self.method.train()
        # Test model
        prediction_y = self.method.test(loaded_test_data['X'])
        # Save 
        self.result.data = {'true_y': loaded_test_data['y'], 'pred_y': prediction_y}
        self.result.save()
        # Evaluation metrics
        return self.evaluate.evaluate(prediction_y, loaded_test_data['y'])