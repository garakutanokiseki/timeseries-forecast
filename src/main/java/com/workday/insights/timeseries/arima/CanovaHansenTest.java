package com.workday.insights.timeseries.arima;

import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

import java.util.Arrays;

import Jama.Matrix;
import Jama.SingularValueDecomposition;

/**
 * Created by amandeeparora on 13/11/18.
 */
public class CanovaHansenTest {

    /***
     * This class is written using by checking these repos
     https://github.com/tgsmith61591/pyramid
     https://github.com/robjhyndman/forecast/blob/master/R/arima.R
     ****/

    private static final double[] criticalValues = {0.4617146, 0.7479655, 1.0007818, 1.2375350, 1.4625240,
            1.6920200, 1.9043096, 2.1169602, 2.3268562, 2.5406922, 2.7391007};


    private CanovaHansenTest() {
    }

    public static int estimateSeasonalDifferencingTerm(double[] ts, int m, int maxD) {
        int D = 0;
        if (D >= maxD) {
            return D;
        }
        boolean dodiff = continueDifferencing(ts, m);
        while (dodiff && D < maxD) {
            D += D+1;
            double[] diffedTs = ArimaAutoFit.dropFirstD(ArimaAutoFit.differencesOfOrderD(ts, D, m), D*m);
            if (isConstant(diffedTs)) {
                return D;
            }
            dodiff = continueDifferencing(diffedTs, m);
        }
        return D;
    }


    private static boolean continueDifferencing(double[] ts, int m) {
        // base case non null ts

        if (ts.length < 2 * m + 5) {
            return false;
        }
        double chstat = sdTest(ts, m);
        if (m <= 12) {
            return chstat > criticalValues[m - 2];
        }
        if (m == 24) {
            return chstat > 5.098624;
        }
        if (m == 52) {
            return chstat > 10.341416;
        }
        if ( m== 365) {
            return chstat > 65.44445;
        }
        return chstat > 0.269 * Math.pow(m, 0.928);
    }


    private static double sdTest(double[] ts, int m) {
        int n = ts.length;
        int[] frec = new int[(m + 1) / 2];
        Arrays.fill(frec, 1);
        int ltrunc = (int) Math.round(m * (Math.pow(n/100.0, 0.25)));
        double[][] R1 = seasDummy(ts, m);

        final OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
        ols.newSampleData(ts, R1);
        double[] residuals = ols.estimateResiduals();

        double[][] Fhataux = new double[n][m-1];
        double[][] Fhat = new double[n][m-1];

        for(int i=0;i<n;i++) {
            for (int j=0; j< m-1; j++) {
                Fhataux[i][j] = R1[i][j] * residuals[i];
                Fhat[i][j] = (i==0 ?  0 : Fhat[i-1][j]) + Fhataux[i][j];
            }
        }

        Matrix FhatMatrix = new Matrix(Fhat);
        Matrix FhatauxMatrix = new Matrix(Fhataux);
        Matrix FhatauxTMatrix = FhatauxMatrix.transpose();

        double[][] Omnw = new double[m-1][m-1];
        Matrix OmnwMatrix = new Matrix(Omnw);
        for (int k=0;k < ltrunc; k++) {
            double wnw = 1 - (k +1.0)/(ltrunc + 1.0);
            OmnwMatrix.plusEquals(FhatauxTMatrix.getMatrix(0, m-2, k+1, n-1).times(FhatauxMatrix.getMatrix(0,n-1-(k+1),0,m-2)).times(wnw));
        }

        Matrix Omfhat = OmnwMatrix.transpose().plus(OmnwMatrix).plus(FhatauxTMatrix.times(FhatauxMatrix)).times(1.0/n);

        /***
         R code
         https://github.com/robjhyndman/forecast/blob/master/R/arima.R#L321

         sq <- seq(1, s - 1, 2)
         frecob <- rep(0, s - 1)
         for (i in 1:length(frec)) {
         if (frec[i] == 1 && i == as.integer(s / 2)) {
         frecob[sq[i]] <- 1
         }
         if (frec[i] == 1 && i < as.integer(s / 2)) {
         frecob[sq[i]] <- frecob[sq[i] + 1] <- 1
         }
         }

         a <- length(which(frecob == 1))
         A <- matrix(0, nrow = s - 1, ncol = a)
         j <- 1
         for (i in 1:(s - 1)) {
         if (frecob[i] == 1) {
         A[i, j] <- 1
         ifelse(frecob[i] == 1, j <- j + 1, j <- j)
         }
         }
         ****/

        double[] frecob = new double[m-1];
        for (int i=0; i < frec.length; i++) {
            if (frec[i] == 1 && i == m/2 - 1) {
                frecob[2*i] = 1;
            }
            if (frec[i] == 1 && i < m/2 - 1) {
                frecob[2*i] = 1;
                frecob[2*i + 1] = 1;
            }
        }

        int a = 0;
        for (int i=0;i<frecob.length;i++) {
            a += (frecob[i] == 1) ? 1: 0;
        }
        Matrix A = new Matrix(m-1, a);
        int j =0;
        for (int i=0;i<m-1;i++) {
            if (frecob[i] == 1) {
                A.set(i, j, 1);
                j +=1;
            }
        }
        Matrix tmp = A.transpose().times(Omfhat).times(A);
        SingularValueDecomposition svd = new SingularValueDecomposition(tmp);
        Matrix sv = svd.getS();

        //# machine min eps
        double eps = Math.pow(2.0,-52.0);
        double min = Double.MAX_VALUE;
        for (int i=0;i<sv.getRowDimension();i++) {
            min = min < sv.get(i,i) ? min : sv.get(i,i);
        }
        if (min < eps) {
            return 0;
        }

        Matrix solved = tmp.solve(Matrix.identity(tmp.getRowDimension(), tmp.getRowDimension()));

        return (1.0/n/n)* solved.times(A.transpose()).times(FhatMatrix.transpose()).times(FhatMatrix).times(A).trace();
    }


    private static double[][] seasDummy(double[] ts, int m) {

        int n = ts.length;
        double[][] fmat = new double[n][m  - 1];
        for (int i = 1; i <= (m + 1)/2 ; i++) {
            for (int j = 0; j < n; j++) {
                if (2*i -1 < m-1) {
                    fmat[j][2 * i - 1] = Math.sin(2 * Math.PI * i * (j + 1) / m);
                }
                if (2*(i-1) < m-1) {
                    fmat[j][2 * (i - 1)] = Math.cos(2 * Math.PI * i * (j + 1) / m);
                }
            }
        }
        return fmat;

    }

    private static boolean isConstant(double[] ts) {
        double val = ts[0];
        for (int i=1;i<ts.length;i++) {
            if (ts[i] != val) {
                return false;
            }
        }
        return true;
    }

}
