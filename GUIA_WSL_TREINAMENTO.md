# Guia Completo: Configuração do Ambiente WSL2 e Treinamento do Detectron2

Este guia documenta o passo a passo completo para configurar o ambiente de desenvolvimento no **WSL2 (Ubuntu)**, instalar o **PyTorch com CUDA**, compilar o **Detectron2**, instalar as dependências específicas (incluindo o OpenCV com *skeletonize*) e executar os scripts de treinamento e validação cruzada.

---

## 📌 Por que utilizar o WSL2 em vez do Windows Nativo?

- **Suporte Nativo a Multiprocessamento (`DataLoader`):** No Windows nativo, `num_workers > 0` frequentemente causa deadlocks ou falhas com o PyTorch/Detectron2. No Linux/WSL2, o carregamento assíncrono com múltiplos workers funciona perfeitamente.
- **Gerenciamento de Memória VRAM:** No Windows, quando a VRAM da GPU esgota, o sistema faz paginação para a RAM convencional de forma lenta (Shared GPU Memory), causando travamentos. No Linux/WSL2, o comportamento do CUDA é muito mais eficiente.
- **Compilação C++/CUDA sem atrito:** Não há necessidade de configurar o MSVC / Visual Studio Build Tools complexos do Windows.

---

## 1. Pré-requisitos no Windows

1. **Driver NVIDIA Atualizado:** Tenha o driver mais recente da sua placa de vídeo instalado no próprio Windows (o WSL2 compartilha o driver do Windows automaticamente).
2. **WSL2 Instalado:**
   Caso ainda não tenha o Ubuntu no WSL, abra o PowerShell do Windows como Administrador e execute:
   ```powershell
   wsl --install -d Ubuntu
   ```
   *(Reinicie o computador se for solicitado).*

---

## 2. Preparação do Ubuntu no WSL2

Abra o terminal do Ubuntu (pelo menu Iniciar ou digitando `wsl` no PowerShell).

### 2.1. Atualizar o sistema e instalar ferramentas de compilação
O Detectron2 compila extensões em C++/CUDA durante a instalação. Execute:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential ninja-build git libgl1 libglib2.0-0
```

### 2.2. Instalar o Miniconda (se ainda não tiver)
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```
> **Atenção:** Feche e reabra o terminal do Ubuntu ou execute `source ~/.bashrc` para ativar o comando `conda`.

---

## 3. Criação do Ambiente Virtual

Crie um ambiente virtual com **Python 3.10**:
```bash
conda create -n detectron2-Seg python=3.10 -y
conda activate detectron2-Seg
```

---

## 4. Instalação do PyTorch com Suporte à GPU

Instale a versão do PyTorch compatível com CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Verificação do CUDA:
Confirme se o PyTorch detecta sua GPU corretamente:
```bash
python -c "import torch; print('CUDA disponível:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Nenhuma')"
```
*Deve exibir `CUDA disponível: True` e o nome da sua placa (ex: NVIDIA GeForce RTX 3050).*

---

## 5. Instalação do Detectron2 e Dependências do Projeto

### 5.1. Instalação do Detectron2
Execute a instalação diretamente do repositório oficial desativando o isolamento de build (para que utilize o PyTorch já instalado no ambiente):
```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
```

### 5.2. Correção Crítica do `setuptools` (`pkg_resources`)
> ⚠️ **Importante:** Versões recentes do `setuptools` (70+) removeram o módulo `pkg_resources`, utilizado internamente pelo `model_zoo` do Detectron2. Caso ocorra `ModuleNotFoundError: No module named 'pkg_resources'`, faça o downgrade:

```bash
pip install "setuptools<70.0.0"
```
*(Ou especificamente: `pip install setuptools==69.5.1`)*

### 5.3. Instalação do OpenCV com Algoritmos Extras (Skeletonize)
Para garantir o suporte a `cv2.ximgproc.thinning` / medição de esqueleto:
```bash
pip install opencv-contrib-python==5.0.0.93
```

### 5.4. Demais Bibliotecas do Projeto
Instale as bibliotecas utilizadas para otimização de hiperparâmetros, manipulação de dados e visualização:
```bash
pip install optuna pandas natsort pycocotools matplotlib plotly
```

---

## 6. Como Acessar o Projeto e Rodar o Treinamento

Os arquivos do seu Windows ficam montados dentro do WSL no diretório `/mnt/c/`.

### 6.1. Navegar até a pasta do projeto
```bash
cd /mnt/c/Users/RIAN/detectron2
```

### 6.2. Ativar o ambiente (se necessário)
```bash
conda activate detectron2-Seg
```

### 6.3. Executar a Validação Cruzada / Treinamento
```bash
python train-maskr-cnn/train-cross_validation.py
```
> **Dica para silenciar warnings:** Se preferir suprimir warnings do terminal via linha de comando, adicione a flag `-W ignore`:
> ```bash
> python -W ignore train-maskr-cnn/train-cross_validation.py
> ```


---

## 7. Dicas de Performance e Monitoramento

1. **Monitorar uso de VRAM da GPU:**
   Em outra aba do terminal do WSL, execute:
   ```bash
   watch -n 1 nvidia-smi
   ```
2. **Número de Workers:**
   No arquivo de treino, você pode manter `WORKERS = 4` (ou `WORKERS = 2`), pois o Linux processa as threads de carregamento de imagem em paralelo sem causar deadlocks.
3. **Controle de Memória para 6GB de VRAM:**
   - Mantenha `cfg.MODEL.BACKBONE.FREEZE_AT = 3` para congelar camadas iniciais da ResNet e economizar memória.
   - Ajuste os tamanhos de imagem (`imgsz` em 256, 512 ou 640) caso a GPU atinja o limite de 6GB.
4. **Abrir o VS Code conectado ao WSL:**
   Dentro da pasta do projeto no WSL, você pode abrir o VS Code com:
   ```bash
   code .
   ```
   Isso instalará a extensão WSL do VS Code, permitindo depurar e editar o código diretamente no ambiente Linux.
