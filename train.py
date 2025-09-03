from torch.utils.tensorboard import SummaryWriter
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, device, epochs,
                lr=0.001,
                early_stopping_patience=5,
                target_accuracy=None,
                checkpoint_dir=None,
                resume_from=None,
                log_dir=None):
    
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    writer = SummaryWriter(log_dir) if log_dir else None

    start_epoch = 0
    best_val_acc = 0.0
    patience_counter = 0

    # Inizializza le liste per salvare metriche
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # Resume da checkpoint
    if resume_from is not None and os.path.isfile(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"Ripreso da checkpoint all'epoca {start_epoch}")

    # Funzione per loggare Loss vs Accuracy come figura
    def log_loss_vs_acc(tag, acc_list, loss_list, epoch):
        fig, ax = plt.subplots()
        ax.plot(acc_list, loss_list, marker='o')
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Loss")
        ax.set_title(tag)
        ax.grid(True, alpha=0.3)
        writer.add_figure(tag, fig, global_step=epoch)
        plt.close(fig)

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0
        correct, total = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Salva metriche
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} - "
              f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

        # TensorBoard logging
        if writer:
            # Grafici classici
            writer.add_scalar("Train/Loss", train_loss, epoch)
            writer.add_scalar("Train/Accuracy", train_acc, epoch)
            writer.add_scalar("Val/Loss", val_loss, epoch)
            writer.add_scalar("Val/Accuracy", val_acc, epoch)
            #Comparazione grafici assieme
            writer.add_scalars("Training Metrics",
                   {"Loss": train_loss, "Accuracy": train_acc}, epoch)
            writer.add_scalars("Validation Metrics",
                   {"Loss": val_loss, "Accuracy": val_acc}, epoch)
            # Grafico XY Loss vs Accuracy
            log_loss_vs_acc("Train/Loss_vs_Accuracy", train_accs, train_losses, epoch)
            log_loss_vs_acc("Val/Loss_vs_Accuracy", val_accs, val_losses, epoch)

        # Save latest model
        if checkpoint_dir:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'patience_counter': patience_counter
            }, os.path.join(checkpoint_dir, 'model_last.pth'))

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0 
            if checkpoint_dir:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'patience_counter': patience_counter
                }, os.path.join(checkpoint_dir, 'model_best.pth'))
            else:
                patience_counter += 1 

        # Early stopping
        if early_stopping_patience and patience_counter >= early_stopping_patience:
            print(f"Early stopping attivato all'epoca {epoch+1}")
            break

        # Stop su target accuracy
        if target_accuracy and val_acc >= target_accuracy:
            print(f"Target accuracy {target_accuracy} raggiunta, stop training")
            break

    if writer:
        writer.close()

    return model, train_losses, val_losses, train_accs, val_accs
