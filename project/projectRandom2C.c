#include <math.h>
#include <stdlib.h>

void projectRandom2C(double *c, double lambda, int nVarsTotal, double *p)
{
    /* Variable Declarations */
    int i, j, nVars, mink, maxk, lowerLen, middleLen, upperLen, k;

    double temp, cand1, cand2, cand3, p_k, p_kP1, p_maxkP1, offset, s1, s2, LHS, RHS, tau;

    /* Check Base Case and Find non-zero variables */
    temp = 0;
    nVars = 0;
    for (i = 0; i < nVarsTotal; i++)
    {
        if (c[i] > 0)
        {
            temp += c[i];
            p[nVars++] = c[i];
        }
    }

    /* Return p = c if sum(c) <= lambda */
    if (temp <= lambda)
    {
        for (i = 0; i < nVarsTotal; i++)
        {
            p[i] = c[i];
        }
        return;
    }

    mink = 0;
    maxk = nVars - 1;
    offset = 0;

    while (1)
    {
        /* Generate 3 random candidates for pivot */
        temp = ((double)rand() / ((double)(RAND_MAX) + (double)(1)));
        cand1 = p[mink + (int)(temp * (1 + maxk - mink))];
        temp = ((double)rand() / ((double)(RAND_MAX) + (double)(1)));
        cand2 = p[mink + (int)(temp * (1 + maxk - mink))];
        temp = ((double)rand() / ((double)(RAND_MAX) + (double)(1)));
        cand3 = p[mink + (int)(temp * (1 + maxk - mink))];

        /* Choose median of candidates as pivot */
        if (cand1 >= cand2 && cand1 <= cand3)
            p_k = cand1;
        else if (cand1 >= cand3 && cand1 <= cand2)
            p_k = cand1;
        else if (cand2 >= cand1 && cand2 <= cand3)
            p_k = cand2;
        else if (cand2 >= cand3 && cand2 <= cand1)
            p_k = cand2;
        else
            p_k = cand3;

        /* Partition Elements in range {mink:maxk} around p_k */
        lowerLen = 0;
        middleLen = 0;
        for (i = mink; i <= maxk; i++)
        {
            if (p[i] > p_k)
            {
                temp = p[i];
                p[i] = p[mink + lowerLen];
                p[mink + lowerLen++] = temp;
                if (p[i] == p_k)
                {
                    temp = p[i];
                    p[i] = p[mink + middleLen];
                    p[mink + middleLen] = temp;
                }
                middleLen++;
            }
            else if (p[i] == p_k)
            {
                temp = p[i];
                p[i] = p[mink + middleLen];
                p[mink + middleLen++] = temp;
            }
        }
        middleLen = middleLen - lowerLen;
        upperLen = maxk - mink - lowerLen - middleLen + 1;

        /* Find out what k value this element corresponds to */
        /* Note: k = (ind+1) since k is a count */
        k = lowerLen + middleLen + mink;

        /* Compute running sum from 1 up to k-1 */
        s1 = offset + p_k * (middleLen - 1);
        for (i = mink; i <= mink + lowerLen - 1; i++)
        {
            s1 += p[i];
        }

        /* Compute Soft-Threshold up to k */
        LHS = s1 - (k - 1) * p_k;

        if (k < nVars)
        {
            if (upperLen == 0)
            {
                p_kP1 = p_maxkP1;
            }
            else
            {
                p_kP1 = 0;
                for (i = mink + lowerLen + middleLen; i <= maxk; i++)
                {
                    if (p[i] > p_kP1)
                    {
                        p_kP1 = p[i];
                    }
                }
            }
        }
        else
        {
            p_kP1 = 0;
        }

        /* Compute Soft-Threshold up to k+1 */
        s2 = s1 + p_k;
        RHS = s2 - k * p_kP1;

        if (lambda >= LHS && (lambda < RHS || upperLen == 0))
        {
            break;
        }

        if (lambda < LHS)
        {
            maxk = k - middleLen;
            p_maxkP1 = p_kP1;
        }
        else
        {
            mink = k;
            offset = s2;
        }
    }

    tau = p_k - (lambda - LHS) / k;
    for (i = 0; i < nVarsTotal; i++)
    {
        if (c[i] - tau > 0)
        {
            p[i] = c[i] - tau;
        }
        else
        {
            p[i] = 0;
        }
    }
}
