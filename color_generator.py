import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from itertools import permutations
from keras.models import load_model


import warnings
warnings.filterwarnings('ignore')






def generator_palette(generator, colors):

    length = len(colors)
    indexs = permutations(range(5), length)

    fullPalettes = []
    img = Image.new("RGB",(256,1),(0,0,0))
    batch = int(256 / 5)
    for k,index in enumerate(indexs):   
        im_fake = np.array(img)
        for count,j in enumerate(index):
            if j == 4:
                im_fake[:,j*batch:,:] = colors[count]
            else:
                im_fake[:,j*batch:(j+1)*batch,:] = colors[count]


        result = generator.predict(np.expand_dims(np.array(im_fake,dtype='float32') / 255 * 2 - 1,0))[0]
        
        TempPalette = []
        for i in range(5):
            # if i not in index:
            mean = np.mean(result[:,i*batch:(i+1)*batch,:],1)[0]
            TempPalette.append(tuple(((mean+1)/2*255).clip(0,255).astype('uint8')))
        # print(TempPalette)
        fullPalettes.append(TempPalette)

    return fullPalettes



def palette2img(palette):
    Img = Image.new("RGB",(100,20),(0,0,0))
    batch = int(100 / len(palette))
    for i, color in enumerate(palette):

        img = Image.new("RGB",(batch,20),color)
        Img.paste(img, (batch*i, 0))

    return Img

if __name__ == '__main__':
    weight_path = sys.argv[1]
    generator = load_model(weight_path)


    nb = random.randint(1,2)
    colors = []
    for i in range(nb):
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        colors.append((r,g,b))
    fullPalettes = generator_palette(generator, colors)
    # print(fullPalettes)



    lens = len(fullPalettes) + 1

    plt.figure(figsize=(5, 10))

    plt.subplot(lens, 1, 1)
    plt.imshow(palette2img(colors))
    plt.axis('off')

    for j in range(1, lens):
        plt.subplot(lens, 1, 1+j)
        plt.imshow(palette2img(fullPalettes[j-1]))
        plt.axis('off')

    plt.show()
