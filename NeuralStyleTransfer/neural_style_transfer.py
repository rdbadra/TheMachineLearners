from PIL import Image
import torchvision.transforms as transforms
import torch
import torchvision.models as models
import torch.nn.functional as F
from torch import optim
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

h, w = 256, 256
mean_rgb = (0.485, 0.456, 0.406)
std_rgb = (0.229, 0.224, 0.225)

transformer = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    transforms.Normalize(mean_rgb, std_rgb)])

content = "data/content.jpeg"
style = "data/style.jpg"

content_img = Image.open(content)
style_img = Image.open(style)

content_tensor = transformer(content_img)
style_tensor = transformer(style_img)

model_vgg = models.vgg19(pretrained=True).features.to(device).eval()

for param in model_vgg.parameters():
    param.requires_grad = False

def get_features(x, model, layers):
    features = {}
    for name, layer in enumerate(model.children()):
        x = layer(x)
        if(str(name) in layers):
            features[layers[str(name)]] = x
    return features

def gram_matrix(input):
    n, c, h, w = input.size()
    features = input.view(n * c, h * w)
    G = torch.mm(features, features.t())
    return G

def get_content_loss(input, target):
    loss = F.mse_loss(input, target)
    return loss

def get_style_loss(pred_features, target_features, feature_layers, content_layer):
    loss = 0
    for i, layer in feature_layers.items():
        if(layer is not content_layer):
            pred_fea = pred_features[layer]
            pred_gram = gram_matrix(pred_fea)
            n, c, h, w = pred_fea.shape
            target_gram = gram_matrix(target_features[layer])
            layer_loss = F.mse_loss(pred_gram, target_gram)
            loss += layer_loss / (n * c * h * w)
    return loss

content_layer = 'conv4_2'

feature_layers = {
    '0': 'conv1_1',
    '5': 'conv2_1',
    '10': 'conv3_1',
    '19': 'conv4_1',
    '21': 'conv4_2',
    '28': 'conv5_1'
}

con_tensor = content_tensor.unsqueeze(0).to(device)
sty_tensor = style_tensor.unsqueeze(0).to(device)

content_features = get_features(con_tensor, model_vgg, feature_layers)
style_features = get_features(sty_tensor, model_vgg, feature_layers)

input_tensor = con_tensor.clone().requires_grad_(True)

optimizer = optim.Adam([input_tensor], lr=0.01)

num_epochs = 301
content_weight = 1e1
style_weight = 1e4

for epoch in range(num_epochs + 1):
    optimizer.zero_grad()
    input_features = get_features(input_tensor, model_vgg, feature_layers)
    content_loss = get_content_loss(input_features[content_layer], content_features[content_layer])
    style_loss = get_style_loss(input_features, style_features, feature_layers, content_layer)
    neural_loss = content_weight * content_loss + style_weight * style_loss
    neural_loss.backward()
    optimizer.step()

    if(epoch % 50 == 0):
        print('epoch {} | Content Loss: {:.2} | Style Loss {:.2}'
            .format(epoch, content_loss, style_loss))

def imgtensor2pil(img_tensor):
    img_tensor_c = img_tensor.clone().detach()
    img_tensor_c *= torch.tensor(std_rgb).view(3, 1, 1)
    img_tensor_c += torch.tensor(mean_rgb).view(3, 1, 1)
    img_tensor_c = img_tensor_c.clamp(0, 1)
    img_pil = to_pil_image(img_tensor_c)
    return img_pil

plt.imshow(imgtensor2pil(input_tensor[0].cpu()))
plt.show()

