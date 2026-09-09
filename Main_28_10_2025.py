import cv2
import numpy as np
import pandas as pd
import warnings
from Model_Faster_RCNN import Model_Faster_RCNN
from Model_YOLOv5 import Model_YOLOv5
from numpy import matlib
import os
import Obj_Seg
from BOA import BOA
from CWO import CWO
from ECO import ECO
from STA import STA
from PROPOSED import PROPOSED
from Global_Vars import Global_vars
from Model_AMSSTD_TYOLOV8_EA import Model_AMSSTD_TYOLOv8_EA
from Model_AT_MHS_AEFNetB7 import Model_AT_MHS_AEFNetB7
from Plot_Results import *
from contextualSeq_AI import contextualSeq_AI
from Model_FPN import Model_FPN
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.style.use('ggplot')
warnings.filterwarnings("ignore")

# Read Dataset
an = 0
if an == 1:
    annot = pd.read_parquet('./Dataset/archive/annot.parquet')
    annot.columns = annot.columns.str.strip()
    img_fns = glob('./Dataset/archive/train_val_images/train_images/*')
    img_fns = img_fns[:5000]
    Original_images = []
    for fn in tqdm(img_fns, desc="Processing images"):
        image_id = os.path.basename(fn).split('.')[0]
        rows = annot.query("image_id == @image_id")
        img = cv2.imread(fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        original_img = cv2.resize(img, (256, 256))
        Original_images.append(original_img)
    Original_images = np.array(Original_images, dtype=np.uint8)
    np.save("Original_Images.npy", Original_images)  # save original Images

# Preprocessing
an = 0
if an == 1:
    loaded_data = np.load("Original_Images.npy", allow_pickle=True)  # Load Original Images
    n_images = loaded_data.shape[0]
    height, width = loaded_data[0].shape[:2]
    # Pre-allocate array
    Preprocessed = np.empty((n_images, height, width), dtype=np.uint8)
    for i in range(n_images):
        img = loaded_data[i]
        print("Img shape:", img.shape)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Gray scale conversion
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10)  # Denoising
        # Sharpen letters
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        enhanced = cv2.filter2D(denoised, -1, kernel)
        Preprocessed[i] = enhanced
    np.save('Preprocessed.npy', Preprocessed)  # save preprocessed Image

# Optimization for Detection
an = 0
if an == 1:
    Feat = np.load('Preprocessed.npy', allow_pickle=True)  # Load preprocessed Image
    Global_vars.Feat = Feat
    Npop = 10
    Chlen = 3  # Hidden neuron count in YoloV8 , Hidden Neuron Count in Attention Layer, Steps per epoch in YoloV8
    xmin = matlib.repmat(np.asarray([5, 5, 100]), Npop, 1)
    xmax = matlib.repmat(np.asarray([255, 255, 500]), Npop, 1)
    fname = Obj_Seg
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("ECO...")
    [bestfit1, fitness1, bestsol1, time1] = ECO(initsol, fname, xmin, xmax,
                                                Max_iter)  # Educational Competition Optimizer

    print("STA...")
    [bestfit2, fitness2, bestsol2, time2] = STA(initsol, fname, xmin, xmax,
                                                Max_iter)  # Supercell Thunderstorm Algorithm

    print("BOA...")
    [bestfit3, fitness3, bestsol3, time3] = BOA(initsol, fname, xmin, xmax, Max_iter)  # Botox Optimization Algorithm

    print("COA...")
    [bestfit4, fitness4, bestsol4, time4] = CWO(initsol, fname, xmin, xmax, Max_iter)  # Carpet Weaver Optimization

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax,
                                                     Max_iter)  # Improved Carpet Weaver Optimization

    BestSol_CLS = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(),
                   bestsol5.squeeze()]
    fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]

    np.save('Fitness.npy', np.asarray(fitness))  # Save the Fitness
    np.save('BestSol.npy', np.asarray(BestSol_CLS))  # save the Bestsol

# Detection
an = 0
if an == 1:
    Images = np.load('Images.npy', allow_pickle=True)  # Load Images
    BestSol = np.load('BestSol.npy', allow_pickle=True)  # Load Bestsol
    Eval_1, Method_1 = Model_Faster_RCNN(Images)  # Faster-RCNN
    Eval_2, Method_2 = Model_FPN(Images)  # FPN
    Eval_3, Method_3 = Model_YOLOv5(Images)  # YOLOv5
    Eval_4, Method_4 = Model_AMSSTD_TYOLOv8_EA(Images)  # AMSSTD-TYOLOv8-EA
    Eval_5, Proposed = Model_AMSSTD_TYOLOv8_EA(Images, sol=BestSol[-1, :])  # proposed
    Annotated = [Method_1, Method_2, Method_3, Method_4, Proposed]
    np.save('Dect_Images.npy', Annotated)  # save Detected Images

# Recognised Text
an = 0
if an == 1:
    Images = np.load('Dect_Images.npy', allow_pickle=True)  # Load Detected Images
    pred = Model_AT_MHS_AEFNetB7(Images)
    np.save('Retrieved_Texts.npy', pred)  # save Retrieved Text

# Post Processing
an = 0
if an == 1:
    Recognised_Text = np.load('Retrieved_Texts.npy', allow_pickle=True)  # save Retrieved Text
    pred = contextualSeq_AI(Recognised_Text)
    np.save('Final.npy', pred)  # save Final

Plot_Conv()  # plot convergence Graph
Plot_kfold()  # plot kfold variation Graph
Plot_Seg_Results()  # plot segmentation Graph
plot_BS_tables()  # plot Batchsize variation Graph
Segmentation_Images()  # plot segmentation Images
Detected_images()  # plot Detected Images
