# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 09:57:08 2018

@author: shen1994
"""

import os
import numpy as np

from utils import get_meta

def useful_image_generate(crop_name, db_name):
    
    imdb_mat_path = crop_name + os.sep + db_name + '.mat'
    
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(imdb_mat_path, db_name)
    
    useful_counter = 0
    images_full_path = []
    images_ages = []
    images_genders = []
    for i in range(len(face_score)):
        
        if face_score[i] < 1.30:
            continue
        if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue
        if ~(0 <= age[i] <= 100):
            continue
        if np.isnan(gender[i]):
            continue
        if not os.path.exists(crop_name + os.sep + str(full_path[i][0])):
            continue
        
        useful_counter += 1
        images_full_path.append(crop_name + os.sep + str(full_path[i][0]))
        images_ages.append(str(age[i]))
        images_genders.append(str(int(gender[i])))

    with open(crop_name + os.sep + db_name + '.csv', 'w') as f:
        
        for i in range(useful_counter):
            
            f.write(images_full_path[i] + ',' \
                    + images_ages[i] + ',' \
                    + images_genders[i] + '\n')    

if __name__ == "__main__":
    
    useful_image_generate('images/imdb_detect', 'imdb')
    print('imdb useful images--->ok')
    useful_image_generate('images/wiki_detect', 'wiki')
    print('wiki useful images--->ok')
        