import torch
import torch.nn as nn
import torch.nn.functional as F

#CNN 모델 정의

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        #합성곱층
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size= 3, padding=1) #입력 이미지 채널 : 1 (흑백), 필터 16개, 필터 크기 3

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding =1)

        #완전 연결층 만들기 :여기는 출력 크기 5로 고정 Linear => 선형회귀함수
        self.fc1 = nn.Linear(64*4*4, 128) 
        self.fc2 = nn.Linear(128, 5) #5개의 타입으로 결과 나옴 
    
    # 순전파 (입력 -> 합성곱 ->풀링->완전연결)
    def forward(self, x):
        # 첫번째 합성곱 + 활성화 함수 + 풀링
        x = F.relu(self.conv1(x)) #ReLU함수 : 입력값이 0보다 작으면 0, 0보다 크면 그대로
        x = F.max_pool2d(x, 2, 2) #2x2 영역으로 설정(최댓값임) #화질 낮춤

        # 두번째 합성곱 처리

        x = F.relu(self.con2(x))
        x = F.max_pool2d(x, 2, 2)

        # 세번째 합성곱 처리

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x,2, 2)

        x = x.view(-1, 64*4*4) #완전연결층 입력 형식에 맞게 조정 (차원 한단계 낮춤 =>1D)

        #완전연결층에 넣기
        x = F.relu(self.fc1(x))
        x= self.fc2(x)

        return(x)