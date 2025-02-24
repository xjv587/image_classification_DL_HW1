from .models import ClassificationLoss, model_factory, save_model, LinearClassifier, MLPClassifier
from .utils import accuracy, load_data, SuperTuxDataset
import torch
import torch.optim as optim
import torch.utils.tensorboard as tb
import tempfile
import os

log_dir = tempfile.mkdtemp()
train_logger = tb.SummaryWriter(log_dir+'/model/train')
valid_logger = tb.SummaryWriter(log_dir+'/model/valid')

path = "C:/Users/jingy/Desktop/DSC394D Deep Learning/homework1/homework1"
tdata_path = os.path.join(path, "data/train")
vdata_path = os.path.join(path, "data/valid")

def train(args):
    model = model_factory[args.model]()
    train_data = load_data(tdata_path, 0, args.batch_size)
    valid_data = load_data(vdata_path, 0, args.batch_size)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        for images, labels in train_data:
            optimizer.zero_grad()
            output = model(images)
            loss = ClassificationLoss()(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += accuracy(output, labels)
        train_loss /= len(train_data)
        train_acc /= len(train_data)


        train_logger.add_scalar('Loss', train_loss, epoch)
        train_logger.add_scalar('Accuracy', train_acc, epoch)

        print(f"Epoch: {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        model.eval()
        valid_loss = 0
        valid_acc = 0
        with torch.no_grad():
            for images, labels in valid_data:
                output = model(images)
                loss = ClassificationLoss()(output, labels)
                valid_loss += loss.item()
                valid_acc += accuracy(output, labels)
        valid_loss /= len(valid_data)
        valid_acc /= len(valid_data)

        valid_logger.add_scalar('Loss', valid_loss, epoch)
        valid_logger.add_scalar('Accuracy', valid_acc, epoch)

        print(f"Validation Loss: {valid_loss:.4f}, Validation Acc: {valid_acc:.4f}")

    save_model(model)


if __name__ == '__main__':
    import argparse


    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--momentum', type=float, default=0.9)

    args = parser.parse_args()
    train(args)
