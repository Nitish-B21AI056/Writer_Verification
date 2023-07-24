from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import joblib
from sklearn.neighbors import KNeighborsClassifier
import pickle

siamese_model_path="SaraSwati_Writes_Final_Model.pth"
knn_model_path='knn_model.pkl'

# Siamses Architectue ---------------------------------------------------------------------------------------------------------------------
class SiameseNetwork(nn.Module):
  def __init__(self):
    super(SiameseNetwork, self).__init__()

    self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=1)
    self.bn1 = nn.BatchNorm2d(num_features=96)
    self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride= 2)
    self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
    self.bn2 = nn.BatchNorm2d(num_features=256)
    self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
    self.dropout1 = nn.Dropout(p=0.3)
    self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
    self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
    self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
    self.dropout2 = nn.Dropout(p=0.3)
    self.fc1= nn.Linear(in_features=108800, out_features=1024)
    self.dropout3 = nn.Dropout(p=0.5)
    self.fc2= nn.Linear(in_features=1024,out_features=128)
    self.relu = nn.ReLU(inplace = True)

  def forward(self, x1, x2):
    x1 = self.conv1(x1)
    x1 = self.relu(x1)
    x1 = self.bn1(x1)
    x1 = self.maxpool1(x1)
    x1 = self.conv2(x1)
    x1 = self.relu(x1)
    x1 = self.bn2(x1)
    x1 = self.maxpool2(x1)
    x1 = self.dropout1(x1)
    x1 = self.conv3(x1)
    x1 = self.relu(x1)
    x1 = self.conv4(x1)
    x1 = self.relu(x1)
    x1 = self.maxpool3(x1)
    x1 = self.dropout2(x1)
    x1 = x1.view(x1.size()[0],-1)
    x1 = self.fc1(x1)
    x1 = self.relu(x1)
    x1 = self.dropout3(x1)
    x1 = self.fc2(x1)

    x2 = self.conv1(x2)
    x2 = self.relu(x2)
    x2 = self.bn1(x2)
    x2 = self.maxpool1(x2)
    x2 = self.conv2(x2)
    x2 = self.relu(x2)
    x2 = self.bn2(x2)
    x2 = self.maxpool2(x2)
    x2 = self.dropout1(x2)
    x2 = self.conv3(x2)
    x2 = self.relu(x2)
    x2 = self.conv4(x2)
    x2 = self.relu(x2)
    x2 = self.maxpool3(x2)
    x2 = self.dropout2(x2)
    x2 = x2.view(x2.size()[0],-1)
    x2 = self.fc1(x2)
    x2 = self.relu(x2)
    x2 = self.dropout3(x2)
    x2 = self.fc2(x2)
    return x1, x2

# Transformation to images -------------------------------------------------------------------------------------------------------------------------
transform=transforms.Compose(
        [transforms.Resize((155, 220)), transforms.ToTensor()])


#Extract output features of siamese model -----------------------------------------------------------------------------------------------------------

def extract_features(model,image1,image2):
    features=[]
    # model.eval()
    # Load the image using PIL
    image1 = Image.open(image1)
    image2 = Image.open(image2)
    # print(sample[0],sample[1])
    # Apply the transformation pipeline
    transformed_image1 = transform(image1)
    transformed_image2 = transform(image2)

    # Move the tensors to the appropriate device
    transformed_image1 = transformed_image1.to(device)
    transformed_image2 = transformed_image2.to(device)

    # Add an extra dimension to match the model input shape
    transformed_image1 = transformed_image1.unsqueeze(0)
    transformed_image2 = transformed_image2.unsqueeze(0)

    # Forward pass through the model
    with torch.no_grad():
        feature=[]
        output1, output2= model(transformed_image1, transformed_image2)
        output1 = output1/torch.norm(output1)
        output2 = output2/torch.norm(output2)
        output1 = output1.cpu().numpy()
        output2 = output2.cpu().numpy()
        feature.append(output1)
        feature.append(output2)
    features.append(feature)
    return features

#Prediction using KNN ---------------------------------------------------------------------------------------------------------------------------------
def knn_predict(test_features):
    knn = joblib.load(knn_model_path)
    predicted_label = knn.predict(test_features)
    probability = knn.predict_proba(test_features)[:,1]
    return predicted_label, probability
print(torch.cuda.is_available())
siamese_model = SiameseNetwork()  # Create an instance of the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    siamese_model.load_state_dict(torch.load(siamese_model_path))
else:
    siamese_model.load_state_dict(torch.load(siamese_model_path, map_location=torch.device('cpu')))

siamese_model.to(device)
siamese_model.eval()
def image_similarity(image1, image2):
    test_features = extract_features(siamese_model, image1, image2)
    test_features = np.array(test_features)
    test_features_reshaped = np.squeeze(test_features)
    test_features_reshaped = np.reshape(test_features_reshaped, test_features.shape)
    test_features_flattened = np.reshape(test_features_reshaped, (test_features.shape[0], -1))

    predicted_test_label, test_probability = knn_predict(np.array(test_features_flattened))
    return predicted_test_label, test_probability[0]
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result1 = None
    result2 = None
    if request.method == 'POST':
        image1 = request.files['image1']
        image2 = request.files['image2']

        if image1 and image2:
            result1, result2 = image_similarity(image1, image2)
            result1 = "Similar" if result1==1 else "Dissimilar"

    return render_template('index.html', result1=result1,result2=result2)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
