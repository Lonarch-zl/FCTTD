# Import standard libraries
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as tnnf
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset
from decomp_fc import TuckerLinearLayer, KruskalLinearLayer
from ignite.metrics import Loss, Accuracy, TopKCategoricalAccuracy
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import ProgressBar

from torch.optim.lr_scheduler import StepLR

# Import utility libraries for model analysis
import click_log

# Import TensorLy for tensor decomposition
import tensorly as tl

# Set TensorLy backend to PyTorch
tl.set_backend('pytorch')

# Configure logging
logger = logging.getLogger(__name__)
click_log.basic_config(logger)

# Constants for dataset and training
DATASET_PATH = "./data"
BATCH_SIZE = 64
NUM_EPOCHS = 100
CIFAR10_DATASET_SIZE = 50000
train_dataset_size = 45000


# Define the VGG16 model
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            # Convolution Blocks (Conv2D + ReLU activations)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Second block
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Continue with additional blocks
        )

        self.classifier = nn.Sequential(
            # Fully Connected Layers
            nn.Linear(in_features=512, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=10),  # Assuming CIFAR-10 (10 classes)
        )

    def forward(self, x):
        # Forward pass through the network
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


# Utility functions for data loading, training, etc.
def load_cifar10_dataset(batch_size, data_path=DATASET_PATH):
    """
    Load and transform the CIFAR10 dataset.

    Args:
    - batch_size: Size of the batches.
    - data_path: Path to the dataset.

    Returns:
    - train_loader: DataLoader for the training data.
    - test_loader: DataLoader for the test data.
    """
    # Data transformation with normalization
    CIFAR10_TRANSFORM_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(15),  # +/- 15 degrees rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    CIFAR10_TRANSFORM_test = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    # Loading the training set
    train_set = datasets.CIFAR10(root=data_path, train=True, download=True, transform=CIFAR10_TRANSFORM_train)
    val_set = datasets.CIFAR10(root=data_path, train=True, download=True, transform=CIFAR10_TRANSFORM_test)
    # 使用random_split获取训练和验证的索引
    train_dataset, val_dataset = random_split(train_set,
                                              [train_dataset_size, CIFAR10_DATASET_SIZE - train_dataset_size])

    # 使用Subset来从每个加载的数据集中获取相应的样本
    train_dataset = Subset(train_set, train_dataset.indices)
    val_dataset = Subset(val_set, val_dataset.indices)
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Loading the test set
    test_set = datasets.CIFAR10(root=data_path, train=False, download=True, transform=CIFAR10_TRANSFORM_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


def decomp_fc(layer, ranks, in_features, out_features):
    # 提取fc1层的权重和偏置
    fc1_weight = layer.weight.data
    fc1_bias = layer.bias.data

    # 定义Tucker秩并创建新的Tucker层
    tucker_layer = TuckerLinearLayer(in_features, out_features, ranks)

    # 使用预训练权重和偏置来设置Tucker层的权重和偏置
    tucker_layer.set_weights(fc1_weight, fc1_bias)

    return tucker_layer


def decomp_core(layer, ranks_tucker, rank_cp, in_features, out_features):
    # 提取fc1层的权重和偏置
    kruskal_layer = KruskalLinearLayer(in_features, out_features, ranks_tucker, rank_cp)

    # 使用预训练权重和偏置来设置Tucker层的权重和偏置
    kruskal_layer.set_weights(layer.core.data, layer.factor_matrices, layer.bias.data)

    return kruskal_layer


def count_parameters_net(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, train_loader, val_loader, epochs, device, metrics):
    """
        Train the VGG16 model with the CIFAR10 dataset.

        Args:
        - model: The neural network model to train.
        - train_loader: DataLoader for the training data.
        - test_loader: DataLoader for the test data.
        - epochs: Number of epochs to train for.
        - device: The device to run the training on ('cuda' or 'cpu').
        """
    # Move model to the specified device
    model.to(device)

    # 加载预训练模型
    model_path = "./model/pretrained_vgg16.pt"
    model.load_state_dict(torch.load(model_path))

    rank_tucker = 8
    rank_kruskal = 64
    model.classifier[0] = decomp_fc(model.classifier[0], [rank_tucker, rank_tucker, rank_tucker, rank_tucker,
                                                          rank_tucker, rank_tucker], [16, 32, 49], [16, 16, 16])

    model.classifier[0] = decomp_core(model.classifier[0],
                                      [rank_tucker, rank_tucker, rank_tucker, rank_tucker, rank_tucker, rank_tucker],
                                      rank_kruskal, [16, 32, 49], [16, 16, 16])

    # 统计线性层的参数数量
    tt_params_count = count_parameters_net(model)
    print(model)
    print("The number of parameters of the linear layer: ", tt_params_count)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=0.0001)

    criterion = tnnf.cross_entropy
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    # 1. 创建学习率调度器
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    @trainer.on(Events.EPOCH_STARTED)
    def update_scheduler(engine):
        model.train()
        scheduler.step()

    def score_function(engine):
        val_loss = engine.state.metrics['cross_entropy_loss']
        return -val_loss

    # 2. 创建模型检查点保存器
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpointer = ModelCheckpoint(checkpoint_dir, 'vgg16',
                                   score_function=score_function,
                                   score_name="val_loss",
                                   n_saved=1,
                                   create_dir=True,
                                   save_as_state_dict=True,
                                   require_empty=False)
    evaluator.add_event_handler(Events.COMPLETED, checkpointer, {'model': model})

    # 3. 为训练过程加入早停机制
    handler = EarlyStopping(patience=25, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, handler)

    # 进度条
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, output_transform=lambda x: {'loss': x})

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        model.eval()
        with torch.no_grad():
            evaluator.run(val_loader)

        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['cross_entropy_loss']
        avg_top5_accuracy = metrics['top5_accuracy']
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        pbar.log_message(
            f"Validation Results - Epoch: {engine.state.epoch} "
            f"Avg accuracy: {avg_accuracy:.4f} "
            f"Avg loss: {avg_loss:.4f} "
            f"Avg top5 accuracy: {avg_top5_accuracy:.4f}"
            f"Current LR: {current_lr:.6f}"
        )

    trainer.run(train_loader, max_epochs=epochs)

    # 训练结束后加载表现最好的模型
    # 列出检查点目录下的所有文件
    model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]

    # 排序以确保获取最新的模型文件（如果有多个文件的话）
    model_files.sort()

    # 如果目录下有模型文件，选择其中的一个（在这种情况下，应该只有一个）
    if model_files:
        best_model_path = os.path.join(checkpoint_dir, model_files[0])
        state_dict = torch.load(best_model_path)
        model.load_state_dict(state_dict)
    else:
        print("No model checkpoint found!")


def evaluate_model(model, test_loader, device, metrics):
    """
    Evaluate the trained model on the test dataset.

    Args:
    - model: The neural network model to evaluate.
    - test_loader: DataLoader for the test data.
    - device: The device to run the evaluation on ('cuda' or 'cpu').
    """
    model.eval()  # Set model to evaluation mode
    test_evaluator = create_supervised_evaluator(model, metrics, device)
    test_evaluator.run(test_loader)
    print(f"On test dataset the best model got: {test_evaluator.state.metrics}")

# Main function to orchestrate the training process
def main():
    # Initialize model, dataloaders, and device
    model = VGG16()
    train_loader, val_loader, test_loader = load_cifar10_dataset(BATCH_SIZE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = {"cross_entropy_loss": Loss(tnnf.cross_entropy), "accuracy": Accuracy(),
               "top5_accuracy": TopKCategoricalAccuracy(k=5, device=device)}
    # Train and evaluate the model
    train_model(model, train_loader, val_loader, test_loader, NUM_EPOCHS, device, metrics)
    evaluate_model(model, test_loader, device, metrics)

if __name__ == "__main__":
    main()
