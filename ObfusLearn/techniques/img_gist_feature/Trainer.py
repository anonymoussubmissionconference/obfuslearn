import os
import csv
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support

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
        for inputs1, labels in self.train_loader:
            inputs1 = inputs1.float()
            if self.device:
                inputs1, labels = inputs1.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs1)
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
            for inputs1,  labels in self.test_loader:
                inputs1 = inputs1.float()
                if self.device:
                    inputs1, labels = inputs1.to(self.device), labels.to(self.device)

                outputs = self.model(inputs1)
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

        # per-class metrics
        classes = np.unique(all_labels)  # or pass your known class ids/order
        prec_c, rec_c, f1_c, support_c = precision_recall_fscore_support(
            all_labels, all_preds, labels=classes, average=None, zero_division=0
        )
        # "Per-class accuracy": usually people just report recall per class.
        # If you really want one-vs-rest accuracy: (TP+TN) / N for each class:
        per_class_acc_ovr = []
        for c in classes:
            tp = np.sum((all_labels == c) & (all_preds == c))
            tn = np.sum((all_labels != c) & (all_preds != c))
            fp = np.sum((all_labels != c) & (all_preds == c))
            fn = np.sum((all_labels == c) & (all_preds != c))
            per_class_acc_ovr.append((tp + tn) / (tp + tn + fp + fn))
        # Note: this is dominated by TN when classes are imbalanced. Often per-class recall is preferred.

        for i, c in enumerate(classes):
            print(f"Class {c}: Prec={prec_c[i]:.4f}, Rec={rec_c[i]:.4f}, F1={f1_c[i]:.4f}, "
                  f"Acc(ovr)={per_class_acc_ovr[i]:.4f}, Support={support_c[i]}")

        # Save summary
        summary_path = os.path.join(self.target_path, "metrics_summary0.15.csv")
        with open(summary_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([precision, recall, f1, accuracy])

        # Save confusion matrix
        matrix_path = os.path.join(self.target_path, "metrics_confusion_matrix0.15.csv")
        with open(matrix_path, mode='a', newline='') as file:
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
