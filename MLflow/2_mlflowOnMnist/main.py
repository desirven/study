import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transfroms

hyperparams = {"learning_rate": 0.0001, "batch_size": 100, "epochs": 5, "num_channels": 2, "seed":1}
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


for _ in range(10):
    hyperparams["num_channels"] += 1
    model = ConvNet(num_channels=hyperparams["num_channels"]).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    import mlflow

        # cli에서 'mlflow server ~~'로 열어둔 서버에 연결
        # 코드내에서 비동기로 열경우 웹ui로 접근이 안되는 문제가 있어서 일단 이렇게 실험
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
        # 없으면 자동으로 experiment 생성해줌
    mlflow.set_experiment("MNIST")

    with mlflow.start_run() as run:
        mlflow.log_params(hyperparams)
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
            mlflow.log_metric('epoch', epoch)
            mlflow.log_metric('cost', -float(avg_cost))

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
                mlflow.log_metric('val_acc', float(correct / total))

        print('Test Accuracy: ', 100. * correct / total, '%')
        mlflow.log_metric('test_acc', float(correct / total))