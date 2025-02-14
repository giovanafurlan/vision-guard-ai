# Vision Guard AI

## Descrição
Vision Guard AI é um sistema de detecção de objetos cortantes em vídeos utilizando redes neurais treinadas com imagens. O projeto possui duas principais etapas:
1. **Treinamento do Modelo**: O modelo é treinado a partir de um dataset contendo imagens de objetos cortantes, gerando um arquivo `best.pt` com os melhores pesos do treinamento.
2. **Identificação em Vídeos**: Com o modelo treinado, um servidor recebe vídeos, processa as imagens e retorna capturas de tela destacando os objetos detectados.

## Como Usar

### 1. Instalação
Certifique-se de ter o Python 3.8+ instalado. Clone o repositório e instale as dependências:
```bash
git clone https://github.com/giovanafurlan/vision-guard-ai.git
cd vision-guard-ai
pip install -r requirements.txt
```

### 2. Executando o Servidor
Inicie o servidor para processar vídeos:
```bash
python server.py
```
Isso disponibilizará uma API que recebe vídeos e retorna prints das detecções.

