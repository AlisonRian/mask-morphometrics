import os, json

def verify_split(path):
    print(f'\n--- Verificando {path} ---')
    images_in_dir = set([f for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')])
    
    json_path = os.path.join(path, 'annotations.json')
    if not os.path.exists(json_path):
        print('annotations.json NÃO ENCONTRADO!')
        return
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    images_in_json = set([img['file_name'].split('/')[-1] for img in data['images']])
    
    print(f'Total de imagens na pasta: {len(images_in_dir)}')
    print(f'Total de imagens no JSON: {len(data["images"])} (Únicas: {len(images_in_json)})')
    print(f'Total de anotações no JSON: {len(data["annotations"])}')
    
    missing_in_json = images_in_dir - images_in_json
    missing_in_dir = images_in_json - images_in_dir
    
    if not missing_in_json and not missing_in_dir:
        print('SUCESSO: As imagens na pasta coincidem perfeitamente com o JSON!')
    else:
        if missing_in_json:
            print(f'ERRO: Existem imagens na pasta que não estão no JSON: {missing_in_json}')
        if missing_in_dir:
            print(f'ERRO: Existem imagens no JSON que não estão na pasta: {missing_in_dir}')

verify_split('train-maskr-cnn/folds/fold1/holdout1/train')
verify_split('train-maskr-cnn/folds/fold1/holdout1/val')
