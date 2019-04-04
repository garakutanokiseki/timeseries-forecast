package com.workday.insights.timeseries.arima;


import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by amandeeparora on 30/10/18.
 */
public class KPSSTest {

    /*
    *
    * This code has been taken from
    * https://github.com/sryza/spark-timeseries/blob/master/src/main/scala/com/cloudera/sparkts/models/ARIMA.scala
    *
    * */

    private KPSSTest() {
    }

    public static final Map<Double, Double> kpssConstantCriticalValues = new HashMap<Double, Double>() {{
        put(0.10, 0.347);
        put(0.05, 0.463);
        put(0.01, 0.739);
        put(0.025, 0.574);
    }};

    public static double evaluate(double[] input) {
        final int n = input.length;

        // labels for regression
        double[][] y = new double[n][1];
        for (double[] row: y) {
            Arrays.fill(row, 1.0);
        }

        //fitting in a regression model
        final OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
        ols.setNoIntercept(true);
        ols.newSampleData(input, y);
        double[] residuals = ols.estimateResiduals();

        double cumulativeSum = 0.0;
        // sum of (square of cumulative sum of errors)
        double s2 = 0.0;
        for (int k = 0; k < n; k++) {
            cumulativeSum += residuals[k];
            s2 += Math.pow(cumulativeSum, 2);
        }
        // long run variance estimate using Newey-West estimator
        // we follow the default lag used in kpss.test in R's tseries package
        int lag = (int) (3 * Math.sqrt(n) / 13);
        double longRunVariance = neweyWestVarianceEstimator(residuals, lag);
        double stat = longRunVariance!= 0 ? (s2 / longRunVariance) / (n * n) : 0;
        return stat;
    }


    private static double neweyWestVarianceEstimator(double[] errors, int lag) {

        int n = errors.length;
        double sumOfTerms = 0.0;
        double cellContrib;
        int i;
        int j;

        i = 1;
        while (i <= lag) {
            j = i;
            cellContrib = 0.0;
            while (j < n) {
                // covariance between values at time t and time t-i
                cellContrib += errors[j] * errors[j - i];
                j += 1;
            }
            // Newey-West weighing (decreasing weight for longer lags)
            sumOfTerms += cellContrib * (1 - (Double.valueOf(i) / (lag + 1)));
            i += 1;
        }
        // we multiply by 2 as we calculated the sum of top row in matrix (lag 0, lag 1), (lag 0, lag 2)
        // etc but we need to also calculate first column (lag 1, lag 0), (lag 2, lag 0) etc
        // we divide by n and obtain a partial estimate of the variance, just missing no lag variance
        double partialEstVar = (sumOfTerms * 2) / n;

        // add no-lag variance
        double variance = partialEstVar;
        for (int k = 0; k < n; k++) {
            variance += Math.pow(errors[k], 2) / n;
        }
        return variance;
    }
}
