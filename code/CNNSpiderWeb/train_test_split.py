# Import the necessary libraries

import os
import random as rd
import shutil

"""## Load Local Modules"""

def get_files_and_shuffle(path):
    files = list(os.listdir(path))
    shuffled = rd.sample(files,len(files))
    return shuffled

def extract_from_list(list,ratio):
    new_list = []
    number_of_files = int(len(list)*ratio)
    for i in range(number_of_files):
        new_list.append(list[i])
    return new_list

"""# Train Test Split - Main"""

featureMethod = 'SpiderWeb'
path = './dataset/'+featureMethod
folders = os.listdir(path+'/train/') #['0', '1', '2', '3', '4', '5', '6', '7']

for folder in folders:
    shuffled = get_files_and_shuffle(path+'/train/'+str(folder)) #List of shuffled list -> name of files of type string
    test_files = extract_from_list(shuffled,0.2)
    for filename in test_files:
        shutil.move(path+'/train/'+str(folder)+'/'+str(filename),path+'/test/'+str(folder)+'/'+str(filename))

number_of_train = len(os.listdir(path+'/train/0/'))
number_of_test = len(os.listdir(path+'/test/0/'))
print("Number of training files per class: ",number_of_train)
print("Number of testing files per class: ",number_of_test)
print("Ratio of test/train: ",(number_of_test/number_of_train))