#ifndef TEST_H
#define TEST_H
#include "Setting.h"
#include "Reader.h"
#include "Corrupt.h"


Triple *negTestList;
extern "C"
void getNegTest() {
    negTestList = (Triple *)calloc(testTotal, sizeof(Triple));
    for (INT i = 0; i < testTotal; i++) {
        negTestList[i] = testList[i];
        negTestList[i].t = corrupt(testList[i].h, testList[i].r);
    }
    FILE* fout = fopen((inPath + "test_neg.txt").c_str(), "w");
    for (INT i = 0; i < testTotal; i++) {
        fprintf(fout, "%ld\t%ld\t%ld\t%ld\n", testList[i].h, testList[i].t, testList[i].r, INT(1));
        fprintf(fout, "%ld\t%ld\t%ld\t%ld\n", negTestList[i].h, negTestList[i].t, negTestList[i].r, INT(-1));
    }
    fclose(fout);
}

Triple *negValidList;
extern "C"
void getNegValid() {
    negValidList = (Triple *)calloc(validTotal, sizeof(Triple));
    for (INT i = 0; i < validTotal; i++) {
        negValidList[i] = validList[i];
        negValidList[i].t = corrupt(validList[i].h, validList[i].r);
    }
    FILE* fout = fopen((inPath + "valid_neg.txt").c_str(), "w");
    for (INT i = 0; i < validTotal; i++) {
        fprintf(fout, "%ld\t%ld\t%ld\t%ld\n", validList[i].h, validList[i].t, validList[i].r, INT(1));
        fprintf(fout, "%ld\t%ld\t%ld\t%ld\n", negValidList[i].h, negValidList[i].t, negValidList[i].r, INT(-1));
    }
    fclose(fout);
        
}

extern "C"
void getTestBatch(INT *ph, INT *pt, INT *pr, INT *nh, INT *nt, INT *nr) {
    getNegTest();
    for (INT i = 0; i < testTotal; i++) {
        ph[i] = testList[i].h;
        pt[i] = testList[i].t;
        pr[i] = testList[i].r;
        nh[i] = negTestList[i].h;
        nt[i] = negTestList[i].t;
        nr[i] = negTestList[i].r;
    }
}

extern "C"
void getValidBatch(INT *ph, INT *pt, INT *pr, INT *nh, INT *nt, INT *nr) {
    getNegValid();
    for (INT i = 0; i < validTotal; i++) {
        ph[i] = validList[i].h;
        pt[i] = validList[i].t;
        pr[i] = validList[i].r;
        nh[i] = negValidList[i].h;
        nt[i] = negValidList[i].t;
        nr[i] = negValidList[i].r;
    }
}

REAL *relThresh;
REAL threshEntire;
extern "C"
void getBestThreshold(REAL *score_pos, REAL *score_neg) {
    REAL interval = 0.01;
    for (INT i = 0; i < 50000; i++) {
        printf("%lf ",score_pos[i]);
    }
    relThresh = (REAL *)calloc(relationTotal, sizeof(REAL));
    REAL min_score, max_score, bestThresh, tmpThresh, bestAcc, tmpAcc;
    INT n_interval, correct, total;
    for (INT r = 0; r < relationTotal; r++) {
        if (validLef[r] == -1) continue;
        printf("relation %ld: left %ld\t right %ld\n", r, validLef[r], validRig[r]);
        total = (validRig[r] - validLef[r] + 1) * 2;
        printf("total:%ld\n",total);
        // min_score = std::min(score_pos[validLef[r]], score_neg[validLef[r]]);
        // max_score = std::max(score_pos[validLef[r]], score_neg[validLef[r]]);
        min_score = score_pos[validLef[r]];
        if (score_neg[validLef[r]] < min_score) min_score = score_neg[validLef[r]];
        max_score = score_pos[validLef[r]];
        if (score_neg[validLef[r]] > max_score) max_score = score_neg[validLef[r]];
        printf("relation %ld: min_score %lf\t max_score %lf\n", r, min_score, max_score);
        for (INT i = validLef[r]+1; i <= validRig[r]; i++) {
            if(score_pos[i] < min_score) min_score = score_pos[i];
            if(score_pos[i] > max_score) max_score = score_pos[i];
            if(score_neg[i] < min_score) min_score = score_neg[i];
            if(score_neg[i] > max_score) max_score = score_neg[i];
        }
        n_interval = INT((max_score - min_score)/interval);
        for (INT i = 0; i <= n_interval; i++) {
            tmpThresh = min_score + i * interval;
            correct = 0;
            for (INT j = validLef[r]; j <= validRig[r]; j++) {
                if (score_pos[j] <= tmpThresh) correct ++;
                if (score_neg[j] > tmpThresh) correct ++;
            }
            tmpAcc = 1.0 * correct / total;
            printf("relation %ld: tmpThresh %lf\t tmpAcc %lf\t bestThresh %lf\t bestAcc %lf\n", r, tmpThresh, tmpAcc, bestAcc, bestThresh);
            if (i == 0) {
                bestThresh = tmpThresh;
                bestAcc = tmpAcc;
            } else if (tmpAcc > bestAcc) {
                bestAcc = tmpAcc;
                bestThresh = tmpThresh;
            }
        }
        relThresh[r] = bestThresh;
        printf("relation %ld: bestThresh is %lf, bestAcc is %lf\n", r, bestThresh, bestAcc);
    }
}

REAL *testAcc;
REAL aveAcc;
extern "C"
void test(REAL *score_pos, REAL *score_neg) {
    testAcc = (REAL *)calloc(relationTotal, sizeof(REAL));
    INT aveCorrect = 0, aveTotal = 0;
    REAL aveAcc;
    for (INT r = 0; r < relationTotal; r++) {
        if (validLef[r] == -1 || testLef[r] ==-1) continue;
        INT correct = 0, total = 0;
        for (INT i = testLef[r]; i <= testRig[r]; i++) {
            if (score_pos[i] <= relThresh[r]) correct++;
            if (score_neg[i] > relThresh[r]) correct++;
            total += 2;
        }
        testAcc[r] = 1.0 * correct / total;
        aveCorrect += correct; 
        aveTotal += total;
        printf("relation %ld: triple classification accuracy is %lf\n", r, testAcc[r]);
    }
    aveAcc = 1.0 * aveCorrect / aveTotal;
    printf("average accuracy is %lf\n", aveAcc);
}

#endif
