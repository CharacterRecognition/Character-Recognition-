import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import warnings
from matplotlib import pylab
from prettytable import PrettyTable
import cv2 as cv
import os
import pylab

warnings.filterwarnings("ignore")


def STATS(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def Plot_Conv():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['Terms', 'ECO-AMSSTD-TYEA', 'STA-AMSSTD-TYEA', 'BOA-AMSSTD-TYEA', 'COA-AMSSTD-TYEA',
                 'RECWO-AMSSTD-TYEA']
    for i in range(Fitness.shape[0]):
        Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
        Conv_Graph = np.zeros((Fitness.shape[1], 5))
        for j in range(len(Algorithm) - 1):
            Conv_Graph[j, :] = STATS(Fitness[i, j, :])
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Report ' + str(i + 1),
              ' --------------------------------------------------')

        print(Table)
        length = np.arange(Fitness.shape[-1])
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='#e50000', linewidth=3, markersize=12, label=Algorithm[1])
        plt.plot(length, Conv_Graph[1, :], color='#0504aa', linewidth=3, markersize=12, label=Algorithm[2])
        plt.plot(length, Conv_Graph[2, :], color='#76cd26', linewidth=3, markersize=12, label=Algorithm[3])
        plt.plot(length, Conv_Graph[3, :], color='#b0054b', linewidth=3, markersize=12, label=Algorithm[4])
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, markersize=12, label=Algorithm[5])
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Result/Convergence.png")
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Convergence Curve')
        plt.show(block=False)
        plt.pause(1)
        plt.close()


def Plot_kfold():
    eval = np.load('Eval_all_KFold.npy', allow_pickle=True)

    Terms = ['Character Accuracy', 'Character Error Rate', 'Word Accuracy', 'Word Error Rate',
             'Lexicon Match Rate', 'Average Processing Time per Image']
    Graph_Terms = [0, 1, 2, 3, 4, 5]
    No_of_task = ['1', '2', '3', '4', '5']
    Classifiers = ['CNN', 'VGG-16', 'Resnet', 'EfficientNet', 'ST-MSAENetB7']
    colours = ['#00BFC4', '#D420CC', '#B5D334', '#9370DB', 'black']
    markers = ['o', 's', '^', 'D', 'X']

    for j in range(len(Graph_Terms)):
        Graph = eval[Graph_Terms[j], :, :]
        MTD_Val = Graph[:, :]

        X = np.arange(MTD_Val.shape[0])

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot lines for each classifier
        for i in range(MTD_Val.shape[-1]):
            ax.plot(X, MTD_Val[:, i],
                    color=colours[i],
                    marker=markers[i],
                    linewidth=2.0,
                    markersize=8,
                    label=Classifiers[i])

        # Formatting
        ax.set_xticks(X)
        ax.set_xticklabels(No_of_task, fontsize=12, fontname="Arial", fontweight='bold')
        ax.set_xlabel('K Fold', fontsize=12, fontweight='bold')
        ax.set_ylabel(Terms[Graph_Terms[j]], fontsize=12, fontweight='bold')
        circle_markers = [Line2D([0], [0], color=colours[i], marker=markers[i],
                                 markersize=8, linewidth=2, label=Classifiers[i])
                          for i in range(len(Classifiers))]
        ax.legend(handles=circle_markers, loc='upper center', bbox_to_anchor=(0.5, -0.1),
                  ncol=5, frameon=False, fontsize=10)
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(f"./Result/TextRecognition_{Terms[Graph_Terms[j]]}_LinePlot.png", dpi=300)
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Task Variation vs ' + Terms[Graph_Terms[j]])
        plt.show(block=False)
        plt.pause(1)
        plt.close()


def Plot_Seg_Results():
    Eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'PSNR', 'MSE', 'Recall', 'Specificity', 'Precision', 'FPR',
             'FNR', 'NPV', 'FDR', 'F1 Score', 'MCC']
    Image = ['Worst', 'Best', 'Mean', 'Median']

    Full = ['TERMS', 'ECO-AMSSTD-TYEA', 'STA-AMSSTD-TYEA', 'BOA-AMSSTD-TYEA', 'COA-AMSSTD-TYEA', 'RECWO-AMSSTD-TYEA',
            'Faster R-CNN', 'FPN', 'YOLOv5', 'MSSTD-TYEA', 'RECWO-AMSSTD-TYEA']
    Graph_terms = [0, 1, 2, 3, 4, 5, 7, 12]
    stats = np.zeros((len(Graph_terms), Eval.shape[-3] + 1, 5))

    Eval_all = Eval
    for k in range(len(Graph_terms)):
        for r in range(5):
            for j in range(Eval_all.shape[-3] + 1):
                if j < Eval_all.shape[-3]:
                    stats[k, j, 0] = np.max(Eval_all[r, j][:, Graph_terms[k] + 4])
                    stats[k, j, 1] = np.min(Eval_all[r, j][:, Graph_terms[k] + 4])
                    stats[k, j, 2] = np.mean(Eval_all[r, j][:, Graph_terms[k] + 4])
                    stats[k, j, 3] = np.median(Eval_all[r, j][:, Graph_terms[k] + 4])
                    stats[k, j, 4] = np.std(Eval_all[r, j][:, Graph_terms[k] + 4])

        alg_prop = stats[k, 4, :]
        stats[k, 9, :] = alg_prop

        Alg_Val = stats[k, :5, 2]
        alg_names = Full[1:6]

        fig, ax = plt.subplots(figsize=(10, 5))
        y = np.arange(len(alg_names))
        colours = ['#0074D9', '#FF851B', '#AAAAAA', '#FFDC00', '#2ECC40']

        # Plot 5 horizontal bars
        bars = ax.barh(y, Alg_Val, color=colours, edgecolor='white')

        # Set y-axis labels
        ax.set_yticks(y)
        ax.set_yticklabels(alg_names, fontsize=10, fontweight='bold')
        ax.set_xlabel(Terms[Graph_terms[k]], fontsize=12, fontweight='bold')
        plt.grid(axis='x', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
        plt.grid(axis='y', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"./Result/TextDetection_{Terms[Graph_terms[k]]}_Alg_Mean.png")
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title(
            f"Horizontal Comparison: {Terms[Graph_terms[k]]}")
        plt.show(block=False)
        plt.pause(1)
        plt.close()
        # plt.show()

        # Method Comparison (Horizontal Bar Plot Style)
        Mtd_Val = stats[k, :5, 2]  # Using MEAN for 5 methods
        method_names = Full[6:11]  # Method labels (5 total)

        fig, ax = plt.subplots(figsize=(10, 5))
        y = np.arange(len(alg_names))
        colours = ['#0074D9', '#FF851B', '#AAAAAA', '#FFDC00', '#2ECC40']

        # Plot 5 horizontal bars
        bars = ax.barh(y, Mtd_Val, color=colours, edgecolor='white')

        # Set y-axis labels
        ax.set_yticks(y)
        ax.set_yticklabels(method_names, fontsize=10, fontweight='bold')

        # Axis labels and title
        ax.set_xlabel(Terms[Graph_terms[k]], fontsize=12, fontweight='bold')
        plt.grid(axis='x', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
        plt.grid(axis='y', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"./Result/TextDetection_{Terms[Graph_terms[k]]}_Mtd_Mean.png")
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title(
            f"Horizontal Comparison: {Terms[Graph_terms[k]]}")
        plt.show(block=False)
        plt.pause(1)
        plt.close()


def plot_BS_tables():
    Eval = np.load('Eval_all_BS.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'PSNR', 'MSE', 'Recall', 'Specificity', 'Precision', 'FPR',
             'FNR', 'NPV', 'FDR', 'F1 Score', 'MCC']
    Graph_terms = [0, 1, 2, 3, 5]

    Full = ['Batch Size', 'ECO-AMSSTD-TYEA', 'STA-AMSSTD-TYEA', 'BOA-AMSSTD-TYEA', 'COA-AMSSTD-TYEA',
            'RECWO-AMSSTD-TYEA',
            'Unet', 'Unet3+', 'ResUnet', 'MSSTD-TYEA', 'RECWO-AMSSTD-TYEA']
    Losses = ['Mse loss', 'Categorical loss', 'Binary loss', 'Center loss', 'SparseCategoricalCrossentropy class']
    BS = ['4', '8', '16', '32', '64']

    Table_stats = np.zeros((Eval.shape[-4], len(Graph_terms), Eval.shape[-3], 5))
    for p in range(Eval.shape[-4]):
        Eval_all = Eval[p]
        for i in range(len(Graph_terms)):
            for j in range(Eval_all.shape[-3]):
                if j < Eval_all.shape[-3]:
                    Table_stats[p, i, j, 0] = np.max(Eval_all[j][:, Graph_terms[i] + 4])
                    Table_stats[p, i, j, 1] = np.min(Eval_all[j][:, Graph_terms[i] + 4])
                    Table_stats[p, i, j, 2] = np.mean(Eval_all[j][:, Graph_terms[i] + 4])
                    Table_stats[p, i, j, 3] = np.median(Eval_all[j][:, Graph_terms[i] + 4])
                    Table_stats[p, i, j, 4] = np.std(Eval_all[j][:, Graph_terms[i] + 4])

    alg_prop = Table_stats[:, :, 4, :]
    Table_stats[:, :, 9, :] = alg_prop
    for t in range(len(Graph_terms)):
        Table = PrettyTable()
        Table.add_column(Full[0], BS)
        for k in range(len(Full) - 6):
            Table.add_column(Full[k + 1], Table_stats[:, t, k, 2])
        print('-------------------------------------------------- Mean Algorithm Comparison_' +
              Terms[Graph_terms[t]],
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Full[0], BS)
        for k in range(5, 10):
            Table.add_column(Full[k + 1], Table_stats[:, t, k, 2])
        print('-------------------------------------------------- Mean Classifier Comparison_' +
              Terms[Graph_terms[t]],
              '--------------------------------------------------')
        print(Table)


def Segmentation_Images():
    Original = np.load('Original_Images.npy', allow_pickle=True)
    Labels = np.load('Preprocessed.npy', allow_pickle=True)
    segmented = np.load('Dect_Images.npy', allow_pickle=True)
    Calssifier = ["Faster R-CNN", "FPN", "YOLOv5", "AMSSTD-TYolov8-EA", "Proposed"]
    Images = [4, 6, 140, 1384, 4984]
    for i in range(len(Images)):
        print(i, len(Images), Images[i])
        Origin = Original[Images[i]]
        Label = Labels[Images[i]]
        Seg_1 = segmented[Images[i]]
        for j in range(1):
            Orig_1 = cv.resize(np.array(Seg_1[j], dtype=np.uint8), (512, 512))
            Orig_2 = cv.resize(np.array(Seg_1[j + 1], dtype=np.uint8), (512, 512))
            Orig_3 = cv.resize(np.array(Seg_1[j + 2], dtype=np.uint8), (512, 512))
            Orig_4 = cv.resize(np.array(Seg_1[j + 3], dtype=np.uint8), (512, 512))
            Orig_5 = cv.resize(np.array(Seg_1[j + 4], dtype=np.uint8), (512, 512))

            plt.suptitle('Detected Images from the Dataset', fontsize=20)

            plt.subplot(3, 3, 1).axis('off')
            plt.imshow(Origin)
            plt.title('Original', fontsize=10)

            plt.subplot(3, 3, 4).axis('off')
            plt.imshow(Orig_1)
            plt.title(Calssifier[0], fontsize=10)

            plt.subplot(3, 3, 5).axis('off')
            plt.imshow(Orig_2)
            plt.title(Calssifier[1], fontsize=10)

            plt.subplot(3, 3, 6).axis('off')
            plt.imshow(Orig_3)
            plt.title(Calssifier[2], fontsize=10)

            plt.subplot(3, 3, 7).axis('off')
            plt.imshow(Orig_4)
            plt.title(Calssifier[3], fontsize=10)

            plt.subplot(3, 3, 9).axis('off')
            plt.imshow(Orig_5)
            plt.title(Calssifier[4], fontsize=10)

            path = "./Result/Image_results/_Compared_Images_%s.png" % (i + 1)
            plt.savefig(path)
            plt.show()

            cv.imwrite('./Result/Image_results/Original_image_' + str(i + 1) + '.png',
                       Origin)
            cv.imwrite(
                './Result/Image_results/Detect_img_' + str(Calssifier[0]) + '_' + str(
                    i + 1) + '.png', Orig_1)
            cv.imwrite(
                './Result/Image_results/Detect_img_' + str(Calssifier[1]) + '_' + str(
                    i + 1) + '.png', Orig_2)
            cv.imwrite(
                './Result/Image_results/Detect_img_' + str(Calssifier[2]) + '_' + str(
                    i + 1) + '.png', Orig_3)
            cv.imwrite(
                './Result/Image_results/Detect_img_' + str(Calssifier[3]) + '_' + str(
                    i + 1) + '.png', Orig_4)
            cv.imwrite(
                './Result/Image_results/Detect_img_' + str(Calssifier[4]) + '_' + str(
                    i + 1) + '.png', Orig_5)
            cv.imwrite('./Result/Image_results/Preprocess_image_' + str(i + 1) + '.png',
                       Label)


def Detected_images():
    Orignal_Image = np.load('Original_Images.npy', allow_pickle=True)
    Preprocessed_Image = np.load('Preprocessed.npy', allow_pickle=True)
    Detected_Image = np.load('Dect_Images.npy', allow_pickle=True)
    Text = np.load('Retrieved_Texts.npy', allow_pickle=True)
    Images = [4, 6, 140, 1747, 4984]
    for i in range(len(Images)):
        print('Retrieved Text of Image :', i + 1)
        Original = cv.resize(Orignal_Image[Images[i]], (512, 512))
        Enhanced = cv.resize(Preprocessed_Image[Images[i]], (512, 512))
        Text_retrived = Text[Images[i]]
        print('Retrived Text : ' + str(Text_retrived))
        Detected = cv.resize(np.asarray(Detected_Image[Images[i]][0]).astype(np.uint8), (512, 512))
        cv.imshow('Orignal Image', Original)
        cv.imshow('Enhanced Image', Enhanced)
        cv.imshow('Detected Image', Detected)
        cv.waitKey(0)
        cv.imwrite('./Result/Detected/Original_Images_' + str(i + 1) + '.png', Original)
        cv.imwrite('./Result/Detected/Enhanced Images_' + str(i + 1) + '.png', Enhanced)
        cv.imwrite('./Result/Detected/Detected Images_' + str(i + 1) + '.png', Detected)
    cv.destroyAllWindows()


if __name__ == '__main__':
    Plot_Conv()
    Plot_kfold()
    Plot_Seg_Results()
    plot_BS_tables()
    Segmentation_Images()
    Detected_images()
