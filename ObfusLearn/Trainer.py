import os
import csv
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, target_path, where_to_save, device=None):
        assert target_path is not None, "Expected target_path to save files"
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.target_path = target_path
        self.where_to_save=where_to_save
        self.device = device

    def train_epoch(self):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        steps = 0
        for (inputs1, inputs2), labels in self.train_loader:
            inputs1, inputs2 = inputs1.float(), inputs2.float()
            if self.device:
                inputs1, inputs2, labels = inputs1.to(self.device), inputs2.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs1, inputs2)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            steps = steps + 1
            print(f"steps {steps}/{len(self.train_loader)} -> "
                  f"Train Loss: {total_loss/total:.4f}, Train Accuracy: {correct / total:.4f}  ")

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def test(self):
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for (inputs1, inputs2), labels in self.test_loader:
                inputs1, inputs2 = inputs1.float(), inputs2.float()
                if self.device:
                    inputs1, inputs2, labels = inputs1.to(self.device), inputs2.to(self.device), labels.to(self.device)

                outputs = self.model(inputs1, inputs2)
                preds = outputs.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")

        # Save summary
        summary_path = os.path.join(self.target_path, "metrics_summary0.1.csv")
        with open(summary_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([precision, recall, f1, accuracy])

        # Save confusion matrix
        matrix_path = os.path.join(self.target_path, "metrics_confusion_matrix0.1.csv")
        with open(matrix_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(conf_matrix)

        return accuracy

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch()
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

            if epoch > self.where_to_save:
                os.makedirs(self.target_path, exist_ok=True)
                modelname =os.path.join(self.target_path, "model" + str(epoch) + ".pth")
                torch.save(self.model.state_dict(), modelname)

        print("Training completed.")
