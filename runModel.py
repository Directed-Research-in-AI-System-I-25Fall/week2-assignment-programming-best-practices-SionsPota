import torch
from transformers import AutoImageProcessor, ResNetModel
import torch
from datasets import load_dataset



# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Using device: {device}')


# 加载mnist数据集
mnist_dataset = load_dataset("ylecun/mnist",trust_remote_code=True)
# mnist_dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
image = mnist_dataset["test"]["image"]
print(mnist_dataset)

# 处理数据
image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
# 加载模型
res_model = ResNetModel.from_pretrained("microsoft/resnet-50")
res_model.to(device)


inputs = image_processor(image, return_tensors="pt")

# 运行模型
with torch.no_grad():
    outputs = res_model(**inputs)

last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)
