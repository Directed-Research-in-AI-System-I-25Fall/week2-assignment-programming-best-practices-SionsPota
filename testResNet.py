from transformers import AutoImageProcessor, ResNetForImageClassification,Pipeline
import torch
from datasets import load_dataset
import datasets
from PIL import Image
import matplotlib.pyplot as plt
plt.ion()

processor = AutoImageProcessor.from_pretrained("arize-ai/resnet-50-fashion-mnist-quality-drift")
model = ResNetForImageClassification.from_pretrained("arize-ai/resnet-50-fashion-mnist-quality-drift")

print(model.config)
print(processor)


dataset = load_dataset("ylecun/mnist",split="test[:100]")
img = dataset['image'][0]

# plt.imsave("image0.png",img)
print(img)

img = img.convert("RGB")
print(img)
# plt.imsave("imageRGB.png",img)

img = processor(img,return_tensors="pt")


print(img)


# Evaluate the model
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for i in range(dataset.num_rows):
        example = dataset[i]
        image = example["image"]
        image = processor(image.convert("RGB"),return_tensors="pt")
        label = example["label"]
        # print(image,label)
        outputs = model(**image)
        _, predicted = torch.max(outputs.logits, 1)
        print(predicted,label)
        total += 1
        correct += (int(predicted) == label)

accuracy = correct / total
print(f'Accuracy: {accuracy * 100:.2f}%')