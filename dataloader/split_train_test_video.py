import os, pickle
import random


class UCF101_splitter():
    def __init__(self, path, split):
        self.path = path
        self.split = split

    def get_action_index(self):
        self.action_label={}
        with open(self.path+'classInd.txt') as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        for line in content:
            label,action = line.split(' ')
            #print label,action
            if action not in self.action_label.keys():
                self.action_label[action]=label

    def subset2(self):
        return self.subset_x(2)

    def subset5(self):
        return self.subset_x(5)

    def subset10(self):
        return self.subset_x(10)

    def subset_x(self, x = 10):
        self.split_video()

        subset_actions = random.sample(list(self.action_label.keys()), x)
        print(subset_actions)
        subset_label = []
        new_label = {}
        label = 1
        for action in subset_actions:
            new_label[self.action_label[action]] = label
            label += 1
            subset_label.append(self.action_label[action])

        subset_train_video = self.subset_dic(self.train_video, subset_label, new_label)
        subset_test_video = self.subset_dic(self.test_video, subset_label, new_label)
        print('==> (Training video, Validation video):(', len(subset_train_video), len(subset_test_video), ')')

        return subset_train_video, subset_test_video

    def subset_dic(self, full_video, subset_label, new_label):
        subset_video = {}
        for video, label in full_video.items():
            if str(label) in subset_label:
                subset_video[video] = new_label[str(label)]
        return subset_video            


    def split_video(self):
        self.get_action_index()
        for path,subdir,files in os.walk(self.path):
            for filename in files:
                if filename.split('.')[0] == 'trainlist'+self.split:
                    train_video = self.file2_dic(self.path+filename)
                if filename.split('.')[0] == 'testlist'+self.split:
                    test_video = self.file2_dic(self.path+filename)
        print('==> (Training video, Validation video):(', len(train_video),len(test_video),')')
        self.train_video = self.name_HandstandPushups(train_video)
        self.test_video = self.name_HandstandPushups(test_video)

        return self.train_video, self.test_video

    def file2_dic(self,fname):
        with open(fname) as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        dic={}
        for line in content:
            #print line
            video = line.split('/',1)[1].split(' ',1)[0]
            key = video.split('_',1)[1].split('.',1)[0]
            label = self.action_label[line.split('/')[0]]   
            dic[key] = int(label)
            #print key,label
        return dic

    def name_HandstandPushups(self,dic):
        dic2 = {}
        for video in dic:
            n,g = video.split('_',1)
            if n == 'HandStandPushups':
                videoname = 'HandstandPushups_'+ g
            else:
                videoname=video
            dic2[videoname] = dic[video]
        return dic2


if __name__ == '__main__':

    path = '../UCF_list/'
    split = '01'
    splitter = UCF101_splitter(path=path,split=split)
    # train_video,test_video = splitter.split_video()
    train_video, test_video = splitter.subset5()
    print(len(train_video),len(test_video))