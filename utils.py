import os
import re
import shutil
import random

def split_train_val_test(input_dir, output_dir, val_ratio=0.15, test_ratio=0.15):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(os.path.join(output_dir, 'train'))
    os.makedirs(os.path.join(output_dir, 'val'))
    os.makedirs(os.path.join(output_dir, 'test'))

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(images)

        total_count = len(images)
        val_count = int(total_count * val_ratio)
        test_count = int(total_count * test_ratio)
        train_count = total_count - val_count - test_count

        train_imgs = images[:train_count]
        val_imgs = images[train_count:train_count + val_count]
        test_imgs = images[train_count + val_count:]

        train_class_path = os.path.join(output_dir, 'train', class_name)
        val_class_path = os.path.join(output_dir, 'val', class_name)
        test_class_path = os.path.join(output_dir, 'test', class_name)

        os.makedirs(train_class_path)
        os.makedirs(val_class_path)
        os.makedirs(test_class_path)

        for img in train_imgs:
            shutil.copy2(os.path.join(class_path, img), os.path.join(train_class_path, img))
        for img in val_imgs:
            shutil.copy2(os.path.join(class_path, img), os.path.join(val_class_path, img))
        for img in test_imgs:
            shutil.copy2(os.path.join(class_path, img), os.path.join(test_class_path, img))


def count_images_per_class(data_dir):
    class_counts = {}
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            count = len([
                f for f in os.listdir(class_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            class_counts[class_name] = count
    return class_counts


def abbrevia_etichetta_clean(label, abbrev_length=3):
    match = re.search(r'_+', label)
    if match:
        idx = match.start()
        abbrev = label[:idx][:abbrev_length].lower()
        rest = label[idx:]
        rest_clean = re.sub(r'_+', '_', rest)
        return abbrev + rest_clean
    else:
        return label[:abbrev_length].lower()


def rinomina_classi_in_dir(original_dir):
    for class_name in os.listdir(original_dir):
        class_path = os.path.join(original_dir, class_name)
        if os.path.isdir(class_path):
            nuova_label = abbrevia_etichetta_clean(class_name)
            nuova_path = os.path.join(original_dir, nuova_label)
            if nuova_label != class_name:
                if not os.path.exists(nuova_path):
                    os.rename(class_path, nuova_path)
                else:
                    print(f"⚠️ Cartella {nuova_label} esiste già, unione non automatica")


def undersample_dataset(original_dir, output_dir, max_per_class=1000):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for class_name in os.listdir(original_dir):
        class_dir = os.path.join(original_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(images)
        selected_images = images[:max_per_class]

        out_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(out_class_dir, exist_ok=True)

        for img_name in selected_images:
            src = os.path.join(class_dir, img_name)
            dst = os.path.join(out_class_dir, img_name)
            shutil.copy2(src, dst)

    print(f"✅ Dataset ridotto salvato in: {output_dir}")