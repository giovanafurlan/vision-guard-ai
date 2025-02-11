import os
import yaml
from ultralytics import YOLO

# Definir o caminho base do dataset
BASE_PATH = r"C:\git\Postech\Hackaton\vision-guard-ai-main"

# Criar dinamicamente o arquivo de configuração do dataset
def create_dataset_yaml():
    dataset_yaml_path = os.path.join(BASE_PATH, "sharp_objects_dataset", "dataset.yaml")
    
    dataset_info = {
        "path": os.path.join(BASE_PATH, "sharp_objects_dataset"),
        "train": os.path.join(BASE_PATH, "sharp_objects_dataset", "images", "train"),
        "val": os.path.join(BASE_PATH, "sharp_objects_dataset", "images", "test"),
        "nc": 1,  # Número de classes
        "names": ["sharp_object"]  # Nome das classes
    }

    with open(dataset_yaml_path, "w") as f:
        yaml.dump(dataset_info, f, default_flow_style=False)

    return dataset_yaml_path

# Função para treinar o modelo YOLOv8
def train_model():
    # Criar o arquivo .yaml dinamicamente
    dataset_yaml = create_dataset_yaml()

    # Carregar o modelo pré-treinado YOLOv8
    model = YOLO("yolov8x.pt")

    # Validar dataset antes do treinamento
    print("Validando a estrutura do dataset...")
    model.val(data=dataset_yaml, plots=True)

    # Iniciar treinamento
    model.train(
        data=dataset_yaml,  
        epochs=40,  
        imgsz=640,  
        batch=16,  
        freeze=0,  
        lr=0.001,  
    )

    # Exportar modelo treinado
    print("Treinamento concluído. Salvando o modelo...")
    model.export(format="onnx")  

    return model

# Executar treinamento
if __name__ == "__main__":
    trained_model = train_model()

    # Testar modelo treinado em uma imagem
    print("Testando o modelo em uma imagem de exemplo...")
    test_image = os.path.join(BASE_PATH, "sharp_objects_dataset", "images", "test", "sample_image.jpg")
    results = trained_model(test_image, conf=0.1)
    results.save()  
    print("Previsões salvas!")
