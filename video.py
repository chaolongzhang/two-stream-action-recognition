import cv2
from spatial_cnn import Spatial_CNN
import torch
from torchvision.transforms import transforms
from PIL import Image
import os


def get_action_index():
    action_label={}
    path = 'D:/MyFile/source/Github/two-stream-action-recognition/UCF_list/'
    with open(path+'classInd.txt') as f:
        content = f.readlines()
        content = [x.strip('\r\n') for x in content]
    f.close()
    for line in content:
        label,action = line.split(' ')
        #print label,action
        if label not in action_label.keys():
            action_label[label]=action
    return action_label

class RecModel(Spatial_CNN):
    
    def __init__(self):
        super(RecModel, self).__init__(1, 0.01, 1, None, 1, False, None, None, None)
        self.resume = 'D:/MyFile/source/Github/two-stream-action-recognition/record/spatial/model_best.pth.tar'
        self.vroot = 'D:/MyFile/dataset/KTH/boxing'
        self.transform = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])
        self.actions = get_action_index()
        self.device = torch.device('cpu')
        self.load()

    def load(self):
        self.build_model()
        self.resume_and_evaluate()

    def predict(self, frame):
        data = self.transform(frame)
        data = data.view(1, 3, 224, 224)
        data = data.to(self.device)
        output = self.model(data)
        preds = output.data.cpu().numpy()
        _, pred = torch.max(output.data, 1)
        action_idx = pred.item()
        print(action_idx, self.actions[str(action_idx)])


    def recoginition(self):
        for file in os.listdir(self.vroot):
            if file.endswith('.avi'):
                fullname = os.path.join(self.vroot, file)
                self.open_video(fullname)

    def open_video(self, fullname):
        print('=' * 10, fullname, '=' * 10)
        cap = cv2.VideoCapture()
        cap.open(fullname)

        while True:
            ret, frame = cap.read()
            if ret == False:
                break
        
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.predict(image)
            cv2.imshow('test', frame)
            cv2.waitKey(20)


if __name__ == "__main__":
   rec = RecModel()
   rec.recoginition()