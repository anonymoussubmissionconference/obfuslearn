import time

from certifi import where
from torch.optim import SGD
from MultiViewVGG11 import *
from Trainer import Trainer
from Multi_View_Loaders import Multi_View_Loaders
from torchvision import transforms


if __name__ == "__main__":
    num_classes = 46
    arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

    # Instantiate model
    model = VGG11WithAPI(arch=arch, num_classes=num_classes)

    # Define transformations
    transform_pipeline = transforms.Compose([
        transforms.Resize((128, 512)),
        transforms.ToTensor()
    ])

    # Setup dataloaders
    datamodule = Multi_View_Loaders(batch_size=32, transform=transform_pipeline)
    train_loader = datamodule.get_train_loader()
    test_loader = datamodule.get_test_loader()

    # Define optimizer and loss
    cost = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)

    target_path = "./models/malfiner-api-256-0.1"
    os.makedirs(target_path, exist_ok=True)
    # Train and evaluate
    where_to_save = 10
    trainer = Trainer(model, train_loader, test_loader, cost, optimizer, target_path=target_path,where_to_save=where_to_save, device=device)
    trainer.train(num_epochs=100)

    start = time.time()
    test_accuracy = trainer.test()
    end = time.time()

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Execution time per sample: {(end - start) / len(test_loader.dataset):.6f} seconds")
