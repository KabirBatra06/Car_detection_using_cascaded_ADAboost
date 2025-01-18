import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import gzip, pickle, pickletools

def get_weak(all_features, labels, weights):

    error = 9999999999999999
    for num_feat in range(all_features.shape[1]):
        features = all_features[:, num_feat]
        sorted_idx = np.argsort(features)
        sorted_feat, sorted_label, sorted_weights = features[sorted_idx], labels[sorted_idx], weights[sorted_idx]

        posW = np.zeros(all_features.shape[0])
        negW = np.zeros(all_features.shape[0])

        posW[sorted_label==1] = sorted_weights[sorted_label==1]
        negW[sorted_label==-1] = sorted_weights[sorted_label==-1]

        polarity1 = np.cumsum(posW) + np.sum(negW) - np.cumsum(negW)
        polarity2 = np.cumsum(negW) + np.sum(posW) - np.cumsum(posW)

        pol1_min_err_idx = np.argmin(polarity1)
        pol2_min_err_idx = np.argmin(polarity2)

        if polarity1[pol1_min_err_idx] < polarity2[pol2_min_err_idx]:
            min_err = polarity1[pol1_min_err_idx]
            min_idx = pol1_min_err_idx
            min_pol = 1
        else:
            min_err = polarity2[pol2_min_err_idx]
            min_idx = pol2_min_err_idx
            min_pol = 2

        if min_err < error:
            error = min_err
            feat_idx = num_feat

            threshold = sorted_feat[min_idx]

            if min_pol == 1:
                pred = np.ones_like(features)
                pred[features < threshold] = -1
            else:
                pred = np.ones_like(features)
                pred[features >= threshold] = -1

    return pred, threshold, feat_idx, min_pol

##CCCCCCCCCCCCCCCCCCCCC
def cascade(features, labels):

    num_ones = np.sum(labels == 1)
    num_zeros = np.sum(labels == -1)

    weights = np.empty(len(labels))
    weights[:num_ones] = 1 / num_ones
    weights[num_ones:] = 1 / num_zeros

    trust = -9999999999999999
    
    for i in range(10):
        weights = weights / np.sum(weights)
        pred, threshold, feat_idx, min_pol = get_weak(features, labels, weights)

        e_t = np.matmul(weights, np.abs(pred - labels)) / 2
        new_trust = (np.log((1-e_t) / e_t)) / 2
        weights = weights * np.exp(-new_trust * labels * pred)

        false_pos = np.sum(pred[num_ones:] == 1) / num_zeros
        false_neg = np.sum(pred[:num_ones] == -1) / num_ones
        
        print(f'false_pos = {false_pos * 100:.3f} %, false_neg = {false_neg * 100:.3f}%')
        
        if new_trust > trust:
            new_pred = pred
            new_false_pos = false_pos
            new_false_neg = false_neg
            trust = new_trust
    
    new_pos_features = features[:num_ones, :]

    wrong_neg = np.where(new_pred[num_ones:] == 1)
    new_neg_features = features[num_ones:, :][wrong_neg]

    combo_features = np.concatenate((new_pos_features, new_neg_features))
    all_labels = np.concatenate((np.ones(len(new_pos_features)), -1*np.ones(len(new_neg_features))))
            
    return combo_features, all_labels, new_false_pos, new_false_neg, threshold, feat_idx, min_pol, trust
    
def get_features(img):
    features = []

    for area in range(2, img.shape[0], 2):
        size = area//2
        pad_img = np.pad(img, ((size, size), (0, 0)), 'constant')
        for i in range(size, (size + img.shape[0])):
            for j in range(pad_img.shape[1]):
                filter = np.concatenate((-1* np.ones(size), np.ones(size))).flatten()
                pix = pad_img[i - size:i+size, j].flatten()
                features.append(np.sum(filter*pix))
    
    for area in range(2, img.shape[1], 2):
        size = area//2
        pad_img = np.pad(img, ((0, 0), (size, size)), 'constant')
        for i in range(pad_img.shape[0]):
            for j in range(size, (size + img.shape[1])):
                filter = np.concatenate((-1* np.ones(size), np.ones(size))).flatten()
                pix = pad_img[i, j-size:j+size].flatten()
                features.append(np.sum(filter*pix))

    return np.array(features)

def main():

    TRAIN_NEG = 'CarDetection/train/negative'
    TRAIN_POS = 'CarDetection/train/positive'
    TEST = 'CarDetection/test'

    train=0
    all_fp = [1]
    all_fn = [1]

    if train:
        all_features = []
        for file in tqdm(os.listdir(TRAIN_NEG)):
            img = cv2.imread(os.path.join(TRAIN_NEG,file), cv2.IMREAD_GRAYSCALE)
            all_features.append(get_features(img))

        np.save("ada_out/neg_feat.npy", np.array(all_features))

        all_features = []
        for file in tqdm(os.listdir(TRAIN_POS)):
            img = cv2.imread(os.path.join(TRAIN_POS,file), cv2.IMREAD_GRAYSCALE)
            all_features.append(get_features(img))

        np.save("ada_out/pos_feat.npy", np.array(all_features))

    pos_features = np.load("ada_out/pos_feat.npy")
    neg_features = np.load("ada_out/neg_feat.npy")

    combo_features = np.concatenate((pos_features, neg_features))
    all_labels = np.concatenate((np.ones(len(pos_features)), -1*np.ones(len(neg_features))))

    for i in range(7):
        combo_features, all_labels, new_false_pos, new_false_neg = cascade(combo_features, all_labels)
        all_fp.append(new_false_pos * all_fp[-1])
        all_fn.append(new_false_neg * all_fn[-1])

        print("FP after Cascade: ", all_fp[-1])
        if all_fp[-1] < 0.000001:
            break

    plt.plot(all_fp[1:], label='False Positive Rate', c='r', marker="o")
    plt.plot(all_fn[1:], label='False Negative Rate', c='g', marker="o")
    plt.legend()
    plt.xlabel('Cascades')
    plt.ylabel('Rate')
    plt.savefig('ada_out/ada_plot.png')

if __name__ == "__main__":
    main()