# Define uma docker image custom que facilita rodar o slowfastsign

FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel

# Set the working directory inside the container
WORKDIR /workspace

# Essa imagem base do pytorch (a que está no FROM) não tem todas as dependências do opencv-python (que é necessário para rodar o slowfastsign). Esses são os pacotes necessários para o opencv-python funcionar.
RUN apt update -y
RUN apt install ffmpeg libsm6 libxext6 -y

# Cria um environment com conda. **TEM** que ser com conda.
RUN conda create -n slowfastvenv
RUN conda install -n slowfastvenv -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia; conda install -n slowfastvenv -y python=3.9

# Favor rodar conda init ao entrar no container para ativar o conda env e depois rodar o restante dos pip install e dependências do SlowFastSign e do ctcdecode por conta própria!
# Esses comandos acima apenas criam um venv do conda compatível com o ctcdecode e o SlowFastSign. 
# Me custou muito da minha sanidade até montar esse venv que consegue rodar as dependências desses dois programas sem quebrar nenhum dos dois.

# Set default command for the container
CMD ["bash"]