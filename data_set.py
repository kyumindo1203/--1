import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
#커스텀 이미지 전처리 시키기
class MyDataSet(Dataset):
    def __init__(self, path):
        self.path = path #파일 경로
        transform = transforms.Compose(
            [
                transforms.Resize((128,128)), #이미지 해상도 설정
                transforms.RandomHorizontalFlip(p=0.5), #50퍼 확률로 좌우 반전 => 128해상도에서 학습 안되면 삭제할 예정
                # transforms.RandomRotation(degrees=30, fill=(0,0,0)), #최대 30도까지 랜덤 회전 => 128해상도에서 학습 안되면 삭제할 예정 , 빈공간은 검은색으로 채우기 => 텐서 오류 발생
                transforms.ToTensor()
            ]
        )
        self.transform = transform

        self.image_paths = [] #이미지 경로들
        self.imgTypes = [] #이미지 종류
        f = open("log.txt", "w",encoding="utf-8") #읽은 파일 로그


        for imgT, imgFolderPath in enumerate(os.listdir(path)):
            # print(imgT, imgFolderPath)
            classPath = os.path.join(path, imgFolderPath) #절대 경로
            print(classPath)
            for imgFile in os.listdir(classPath):
                f.write(imgFile+"\n")
                self.image_paths.append(os.path.join(classPath,imgFile)) #사진의 절대경로
                self.imgTypes.append(imgT) #이미지 종류
        f.close()

    def __len__(self):
        return len(self.image_paths) # 입력된 이미지 갯수
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        imgType = self.imgTypes[idx]
        image = Image.open(img_path).convert("RGB") #.convert("RGB") => RGB형식으로 파일 열기
        
        if self.transform:
            image = self.transform(image) #텐서로 전처리
        return image, imgType#, img_path 
    

if __name__ == '__main__':
    path = os.path.dirname(os.path.abspath(os.path.dirname(__file__))) #상위폴더

    #전처리 설정
    # transform = transforms.Compose(
    #     [
    #         transforms.Resize((128,128)), #이미지 해상도 설정
    #         transforms.RandomHorizontalFlip(p=0.5), #50퍼 확률로 좌우 반전 => 128해상도에서 학습 안되면 삭제할 예정
    #         # transforms.RandomRotation(degrees=30, fill=(0,0,0)), #최대 30도까지 랜덤 회전 => 128해상도에서 학습 안되면 삭제할 예정 , 빈공간은 검은색으로 채우기 => 텐서 오류 발생
    #         transforms.ToTensor()
    #     ]
    # )

    
    # d = open("Tensor_info.txt","w",encoding="utf-8")
    a = MyDataSet(f"{path}\\DataSets\\images")
    # for i in range(a.__len__()):
    #     Img, Type, img_path= a.__getitem__(i)
    #     d.write(f"(img:{Img}, type:{Type}, path:{img_path})\n")
    DataA = DataLoader(a, batch_size=32, shuffle=True) # 배치 설정
    for i, j in DataA:
    #     # d.write(f"(img:{Img}, type:{Type}, path:{img_path})\n")
    #     # d.write(f"i:{i}, j:{j}, k:{k}\n\n")
        print(f"i : {i.shape}, j : {j.shape}")

    # d.close()

else:
    path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    a = MyDataSet(f"{path}\\DataSets\\images")
    DataA = DataLoader(a, batch_size=32, shuffle=True)