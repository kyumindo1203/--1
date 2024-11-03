import torch
import torch.nn as nn
import torch.nn.functional as F
from CNN_model import CNN
from data_set import trainA, valA, BS
import os
import time
import datetime
if __name__ == "__main__":
    f = open("TraningLog.txt","w",encoding='utf-8')
    # path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

    # if torch.cuda.is_available():
    #     print("CUDA is available! Using GPU!!")
    #     device = torch.device("cuda")
    # else:
    #     print("CUDA is not available. Using CPU.")
    #     device = torch.device("cpu")

    start_time = datetime.datetime.now()

    model = CNN(BS=BS)
    # model.to(device)
    f.write(f"<Model Training Start at {start_time}>\n\n")
    f.write(f"Model Load as {model}\n")

    criterion = nn.CrossEntropyLoss() #손실함수 사용
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001) #가중치 조절옵티마이저 설정

    f.write(f"LossFunc&optimizer Loaded")
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    #학습(에포크, 손실계산)
    num_ep = 10 #에포크 크기
    f.write(f"Epoch Size : {num_ep}--Training Start\n")
    for epoch in range(num_ep):
        for img, imgType in trainA:
            # print(f"input : {img.shape}, output : {imgType.shape}")
            #img가 입력, imgType이 정답
            
            #데이터 GPU로 이동
            # img, imgType = img.to(device), imgType.to(device)

            outputs = model(img) # 모델의 예측 값
            loss = criterion(outputs, imgType) #손실 계산; 정답가 얼마나 거리가 먼지(확률분포)
            # print(loss)

            #역전파, 가중치 계산
            loss.backward() #역전파
            optimizer.step()
            optimizer.zero_grad() #기울기 초기화
        print(f"[training finished] Lastest Loss:{loss.item():.4f}")
        
        #평가    
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():#기울기 계산 비활성화
            for img, imgType in valA:

                #데이터 GPU로 이동
                # img, imgType = img.to(device), imgType.to(device)

                outputs = model(img)
                loss = criterion(outputs, imgType)

                val_loss += loss.item()

                _, predicted = torch.max(outputs,1) # 예측 클래스 선택
                correct += (predicted == imgType).sum().item() #정답 수 

        avrage_val_loss = val_loss/len(valA)
        accuracy = correct/len(valA.dataset)
        print(f"Validation Loss: {avrage_val_loss:.4f}, Accuracy: {accuracy:.4f}")
        print(f"Epoch [{epoch+1}/{num_ep}], Loss:{loss.item():.4f}\n")

        f.write(f"Validation Loss: {avrage_val_loss:.4f}, Accuracy: {accuracy:.4f}|")
        f.write(f"Epoch [{epoch+1}/{num_ep}], Loss:{loss.item():.4f}\n")
    f.write(f"\nTraining_Done! Final Accuracy: {accuracy*100:.4f}%")
    now= datetime.datetime.now().strftime("%d%H%M%S%f")
    print(now)
    model_path = f'model-V{now}.cnn'
    torch.save(model, model_path)
    f.write(f"\nModel Saved at {model_path}")
    f.close()