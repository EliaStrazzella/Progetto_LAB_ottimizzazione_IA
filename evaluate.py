import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import os
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
from collections import defaultdict

def plot_training(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names)

def plot_confusion_matrix(cm, class_names, normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=fmt, xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def plot_distribution(distribution_dict, title):
    classes = list(distribution_dict.keys())
    counts = list(distribution_dict.values())

    plt.figure(figsize=(12, 6))
    plt.bar(classes, counts, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Classi')
    plt.ylabel('Numero di Immagini')
    plt.title(title)
    plt.tight_layout()
    plt.show()
def visualize_sample_images(dataset_path, class_name=None, num_images=2):
 
    # Se class_name non è specificato, prendi la prima classe trovata
    class_dir = os.path.join(dataset_path, class_name) if class_name else None
    if not class_dir or not os.path.exists(class_dir):
        class_list = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        if not class_list:
            print("❌ Nessuna classe trovata nel dataset.")
            return
        class_name = class_list[0]
        class_dir = os.path.join(dataset_path, class_name)

    # Prendi immagini
    image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:num_images]

    if not image_files:
        print(f"❌ Nessuna immagine trovata nella classe: {class_name}")
        return

    # Visualizza immagini
    fig, axes = plt.subplots(1, len(image_files), figsize=(5 * len(image_files), 5))
    if len(image_files) == 1:
        axes = [axes]  # per compatibilità

    for ax, img_file in zip(axes, image_files):
        img_path = os.path.join(class_dir, img_file)
        image = Image.open(img_path)
        ax.imshow(image)
        ax.set_title(f"{class_name}\n{img_file}", fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
def plot_leaf_type_pie_chart(dataset_dir):
   
    type_counts = defaultdict(int)

    prefix_map = {
        "tom": "Tomato",
        "pep": "Pepper",
        "pot": "Potato"
    }

    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        # Estrai il prefisso
        prefix = class_name.split('_')[0].lower()
        if prefix not in prefix_map:
            continue  # ignora classi non mappate

        leaf_type = prefix_map[prefix]

        # Conta immagini valide
        image_count = len([
            f for f in os.listdir(class_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        type_counts[leaf_type] += image_count

    # Se vuoto
    if not type_counts:
        print("❌ Nessuna immagine trovata per Tomato, Potato o Pepper.")
        return

    # Plot pie chart
    labels = list(type_counts.keys())
    sizes = list(type_counts.values())

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Distribuzione tipi di foglie (Tomato, Potato, Pepper)")
    plt.axis('equal')
    plt.show()