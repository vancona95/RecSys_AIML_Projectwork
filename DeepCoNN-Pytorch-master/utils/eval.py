import torch
from data_reader import get_train_dev_test_data
from train_helper import load_model, eval_model, get_data_loader

train_data, dev_data, test_data = get_train_dev_test_data()

model = load_model("model/checkpoints/DeepCoNN_20221106005252.pt")
model.to(model.config.device)
loss = torch.nn.MSELoss()
data_iter = get_data_loader(test_data, model.config)
result, list_predicts = eval_model(model, data_iter, loss)

print(eval_model(model, data_iter, loss))
