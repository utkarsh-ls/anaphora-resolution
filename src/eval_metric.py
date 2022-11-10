import numpy as np
import copy
from itertools import combinations


def MUC_score(K, R):
    muc_score = 0
    for i in range(len(K)):
        num = len(K[i])
        intersects = []
        left_over = copy.deepcopy(K[i])
        for j in range(len(R)):
            tmp = np.intersect1d(R[j], K[i])
            left_over = np.intersect1d(K[i], np.setxor1d(left_over, R[j]))
            if len(tmp) > 0:
                intersects.append(tmp)

        num -= len(intersects) + len(np.intersect1d(left_over, K[i]))
        muc_score += num

    denom = 1e-6
    for i in range(len(K)):
        denom += len(K[i]) - 1

    muc_score /= denom

    return muc_score


def B3_score(K, R):
    b3_score = 0
    for i in range(len(K)):
        for j in range(len(R)):
            b3_score += (len(np.intersect1d(K[i], R[j])) ** 2) / (1e-6 + len(K[i]))
    denom = 1e-6
    for i in range(len(K)):
        denom += len(K[i])

    return b3_score / denom


def BLANC_score(K, R):
    K_pairs = []
    for i in range(len(K)):
        K_pairs.append(list(combinations(K[i], 2)))

    K_list = []
    for i in range(len(K_pairs)):
        K_list += K_pairs[i]

    R_pairs = []
    for i in range(len(R)):
        R_pairs.append(list(combinations(R[i], 2)))

    R_list = []
    for i in range(len(R_pairs)):
        R_list += R_pairs[i]

    NK = []
    for i in range(len(K)):
        for j in range(i, len(K)):
            if j != i:
                NK += [(x, y) for x in K[i] for y in K[j]]

    NR = []
    for i in range(len(R)):
        for j in range(i, len(R)):
            if j != i:
                NR += [(x, y) for x in R[i] for y in R[j]]

    rc_intersect = []
    for i in range(len(K_list)):
        if K_list[i] in R_list:
            rc_intersect.append(K_list[i])

    pc_intersect = len(rc_intersect) / (1e-6 + len(R_list))
    rc_intersect = len(rc_intersect) / (1e-6 + len(K_list))

    nrc_intersect = []
    for i in range(len(NK)):
        if NK[i] in NR:
            nrc_intersect.append(NK[i])

    npc_intersect = len(nrc_intersect) / (1e-6 + len(NR))
    nrc_intersect = len(nrc_intersect) / (1e-6 + len(NK))

    return (f1(rc_intersect, pc_intersect) + f1(npc_intersect, nrc_intersect)) / 2


def f1(recall, precision):
    return 2 * precision * recall / (1e-6 + precision + recall)


def get_all_scores(K, R):
    return [
        f1(MUC_score(K, R), MUC_score(R, K)),
        f1(B3_score(K, R), B3_score(R, K)),
        f1(BLANC_score(K, R), BLANC_score(R, K)),
    ]


def get_accs_mention(men_gt, men_pred, sentence_len):
    men_gt = [i for i in men_gt if i < sentence_len]
    men_pred = [i for i in men_pred if i < sentence_len]
    gt_mask = np.zeros(sentence_len)
    gt_mask[men_gt] = 1
    pred_mask = np.zeros(sentence_len)
    pred_mask[men_pred] = 1

    return get_accs_from_mask(gt_mask, pred_mask)


def get_accs_pairscore(is_pair_gt, is_pair_pred):
    return get_accs_from_mask(np.asarray(is_pair_gt), np.asarray(is_pair_pred))


def get_accs_from_mask(gt_mask, pred_mask):
    tp = (gt_mask * pred_mask).sum()
    tn = ((1 - gt_mask) * (1 - pred_mask)).sum()
    fp = ((1 - gt_mask) * pred_mask).sum()
    fn = (gt_mask * (1 - pred_mask)).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return [accuracy, precision, recall, f1]


if __name__ == "__main__":
    K = np.array([[0, 1, 2], [3, 4, 5, 6]])
    R = np.array([[0, 1], [2, 3], [5, 6, 7, 8]])

    recall_muc = MUC_score(K, R)
    precision_muc = MUC_score(R, K)

    print("MUC F1 score:", f1(recall_muc, precision_muc))

    recall_b3 = B3_score(K, R)
    precision_b3 = B3_score(R, K)

    print("B3 F1 score:", f1(recall_b3, precision_b3))

    print("BLANC score:", BLANC_score(K, R))
