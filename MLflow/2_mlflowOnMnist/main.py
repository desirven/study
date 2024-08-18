import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transfroms

hyperparams = {"learning_rate": 0.0001, "batch_size": 100, "epochs": 5, "num_channels": 10, "seed":1}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(hyperparams["seed"])
if device == 'cuda':
    torch.cuda.manual_seed_all(hyperparams["seed"])
print(device + " is available")


train_set = torchvision.datasets.MNIST(
    root='./data/MNIST',
    train=True,
    download=True,
    transform=transfroms.Compose([
        transfroms.ToTensor()
    ])
)
test_set = torchvision.datasets.MNIST(
    root='./data/MNIST',
    train=False,
    download=True,
    transform=transfroms.Compose([
        transfroms.ToTensor()
    ])
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=hyperparams["batch_size"])
test_loader = torch.utils.data.DataLoader(test_set, batch_size=hyperparams["batch_size"])

class ConvNet(nn.Module):
    def __init__(self, num_channels=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=5)
        self.conv2 = nn.Conv2d(num_channels, num_channels*2, kernel_size=5)
        self.drop2D = nn.Dropout2d(p=0.25, inplace=False)
        self.mp = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(num_channels*32, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = self.drop2D(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x)

model = ConvNet(num_channels=hyperparams["num_channels"]).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

import mlflow

    # cli에서 'mlflow server ~~'로 열어둔 서버에 연결
mlflow.set_tracking_uri("http://127.0.0.1:5001")
    # create를 안해도 set에서 자동으로 생성해주지만 artifact_location을 지정하려면 사용해야한다.
# mlflow.create_experiment(experiment_name, artifact_location="s3://your-bucket")
mlflow.set_experiment("MNIST")

with mlflow.start_run() as run:
        # 파라미터 로깅
    mlflow.log_params(hyperparams)
        # 태그 로깅
    mlflow.set_tags({"version":"1.0.0", "topic":"change num of conv channels"})
        # 아티펙트 로깅
    from PIL import Image
    samples_testset = [test_set[i][0].squeeze() for i in torch.randint(len(test_set), size=(16,))]
    sample_image = Image.new('L', (128, 128))
    for i, sample in enumerate(samples_testset):
        img = Image.fromarray((sample.numpy() * 255).astype('uint8'), mode='L')
        x = (i % 4) * 32
        y = (i // 4) * 32
        sample_image.paste(img, (x, y))
    sample_image.save("./samples.png")
    mlflow.log_artifact("./samples.png")

    print("start training")
    # train
    for epoch in range(hyperparams["epochs"]):
        avg_cost = 0
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            hypothesis = model(data)
            cost = criterion(hypothesis, target)
            cost.backward()
            optimizer.step()
            avg_cost += cost / len(train_loader)
        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
    # 메트릭 저장
        mlflow.log_metric('cost', -float(avg_cost), step=epoch+1)

    # test
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0

            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)
                out = model(data)
                preds = torch.max(out.data, 1)[1]
                total += len(target)
                correct += (preds == target).sum().item()
            print('Val Accuracy: ', 100. * correct / total, '%')
            mlflow.log_metric('val_acc', float(correct / total), step=epoch+1)

# MLFlow models
    from mlflow.models.signature import infer_signature
    sample_data = next(iter(train_loader))[0][0].unsqueeze(0)
    sample_data_np = sample_data.numpy()
    signature = infer_signature(sample_data_np, model(sample_data).detach().numpy())
    mlflow.pytorch.log_model(model, "model", signature=signature)

model_uri = f"runs:/{run.info.run_id}/model"
loaded_model = mlflow.pytorch.load_model(model_uri)

with torch.no_grad():
    correct = 0
    total = 0

    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        out = loaded_model(data)
        preds = torch.max(out.data, 1)[1]
        total += len(target)
        correct += (preds == target).sum().item()

print('Test Accuracy: ', 100. * correct / total, '%')
mlflow.log_metric('test_acc', float(correct / total))
