import torch
from CNN_model import CNN
from data_set import MyDataSet
from torch.utils.data import DataLoader
import torch.nn as nn
import os
if __name__ == '__main__':

    path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    testSets = MyDataSet(f"{path}\\DataSets\\testIMG", limit=200)

    testA = DataLoader(testSets, batch_size=32,drop_last=True)
    model = CNN(BS=32)

    model_path = 'model-V04040220487279.cnn'
    device = "cpu"
    model = torch.load(model_path, map_location=device)


    criterion = nn.CrossEntropyLoss() #손실함수 사용
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for img, imgType in testA:
            outputs = model(img)
            test_loss += criterion(outputs, imgType).item() #손실 더하기 .item()은 텐서를 숫자로 바꿔줌
            _, predicted = torch.max(outputs, 1) 
            correct += (predicted == imgType).sum().item()

    avrage_test_loss = test_loss/len(testA)
    accuracy = correct/len(testA.dataset)
    print(f"Validation Loss: {avrage_test_loss:.4f}, Accuracy: {accuracy:.4f} in TestSets")