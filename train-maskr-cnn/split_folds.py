import json
import os
import shutil
import random
from natsort import natsorted

batches = [f"L{i}" for i in range(1, 9)]
test_batch = batches[0]
train_batches = [batch for batch in batches if batch != test_batch]

def split_folds(dataset_path, json_path, output_path):
    """ Usado unicamente para separar os 8 lotes e gerar os arquivos json com as anotações corretos """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i in range(0,8):
        batch_path = os.path.join(output_path,f"L{i+1}")
        if not os.path.exists(batch_path):
            os.makedirs(batch_path)

    for i in os.listdir(dataset_path):
        if i.endswith('.jpg') or i.endswith('.png'):   
            subs = i.split('-')
            batch = subs[0]
            destiny = os.path.join(output_path,batch,i)
            file_path = os.path.join(dataset_path,i)
            shutil.copy(file_path,destiny)
    
    with open(json_path, "r") as file:
        data = json.load(file)

    for i in range(1, 9):
        fold_i_img = [file for file in data['images'] if f'L{i}' in file['file_name']]
        annotation_i = [anot for anot in data['annotations'] if anot['image_id'] in [img['id'] for img in fold_i_img]]

        json_data = data.copy()
        json_data['images'] = fold_i_img
        json_data['annotations'] = annotation_i
        with open(os.path.join(output_path, "L"+str(i), "annotations.json"), "w") as file:
            json.dump(json_data, file)

def split_to_holdout_train(dataset_path:str, train_batches:list[str], fold: int) -> dict[str, str]:
    random.seed(42+fold)
    holdout_path = f"./train-maskr-cnn/folds/fold{fold}/holdout{fold}"
    for split in ['train', 'val']:
        os.makedirs(os.path.join(holdout_path, split), exist_ok=True)

    train_json_data = None
    val_json_data = None

    for b in os.listdir(dataset_path):
        if b in train_batches:
            images_path = os.path.join(dataset_path, b)
            all_imgs = natsorted(f for f in os.listdir(images_path) if f.endswith(('.jpg','.png')))
            annotation_file = [f for f in os.listdir(images_path) if f.endswith('.json')][0]
            total_imgs = len(all_imgs)

            val_imgs = set(random.sample(all_imgs, int(0.20*total_imgs)))
            print(f"Lote {b}: {len(val_imgs)} imagens sorteadas para validação de {total_imgs} totais.")

            for img in all_imgs:
                dest_folder = 'val' if img in val_imgs else 'train'
                src_img = os.path.join(images_path, img)
                dst_img = os.path.join(holdout_path, dest_folder, img) # Garantindo que o destino receba o nome do arquivo
                shutil.copy(src_img, dst_img)

            with open(os.path.join(images_path, annotation_file), "r") as file:
                data = json.load(file)

            if train_json_data is None:
                train_json_data = {k: v for k, v in data.items() if k not in ['images', 'annotations']}
                train_json_data['images'] = []
                train_json_data['annotations'] = []
                
                val_json_data = {k: v for k, v in data.items() if k not in ['images', 'annotations']}
                val_json_data['images'] = []
                val_json_data['annotations'] = []

            prefix_to_disk = {f.split('.rf.')[0]: f for f in all_imgs}

            b_train_imgs = []
            b_val_imgs = []

            for img in data['images']:
                json_basename = img['file_name'].split('/')[-1]
                prefix = json_basename.split('.rf.')[0]

                if prefix in prefix_to_disk:
                    disk_filename = prefix_to_disk[prefix]
                    img['file_name'] = disk_filename

                    if disk_filename in val_imgs:
                        b_val_imgs.append(img)
                    else:
                        b_train_imgs.append(img)
                else:
                    print(f"Aviso: imagem {json_basename} presente no JSON não tem correspondente na pasta.")

            train_img_ids = set([img['id'] for img in b_train_imgs])
            val_img_ids = set([img['id'] for img in b_val_imgs])

            b_train_annotations = [anot for anot in data['annotations'] if anot['image_id'] in train_img_ids]
            b_val_annotations = [anot for anot in data['annotations'] if anot['image_id'] in val_img_ids]

            train_json_data['images'].extend(b_train_imgs)
            train_json_data['annotations'].extend(b_train_annotations)

            val_json_data['images'].extend(b_val_imgs)
            val_json_data['annotations'].extend(b_val_annotations)

    with open(os.path.join(holdout_path, 'train', "annotations.json"), "w") as file:
        json.dump(train_json_data, file)

    with open(os.path.join(holdout_path, 'val', "annotations.json"), "w") as file:
        json.dump(val_json_data, file)
        
    output = {
        'TRAIN_DATA_DIR':os.path.join(holdout_path, 'train'),
        'VAL_DATA_DIR':os.path.join(holdout_path, 'val'),
        'TRAIN_JSON_DIR':os.path.join(holdout_path, 'train', 'annotations.json'),
        'VAL_JSON_DIR':os.path.join(holdout_path, 'val', 'annotations.json')
    }
    return output

# split_folds("C:/Users/RIAN/detectron2/train-maskr-cnn/Dataset - COCO_corrigido",
#  "C:/Users/RIAN/detectron2/train-maskr-cnn/Dataset - COCO_corrigido/instances_default.json",
#   "C:/Users/RIAN/detectron2/train-maskr-cnn/Dataset_splitted")
# split_to_holdout_train("C:/Users/RIAN/detectron2/train-maskr-cnn/Dataset_splitted", train_batches, 1)