package com.workday.insights.timeseries.arima;

import com.spr.intuition.ds.Quadruple;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by amandeeparora on 31/10/18.
 */
public class ArimaAutoFit {

    /***
     * This class is written using by checking these three repos
    https://github.com/sryza/spark-timeseries/blob/master/src/main/scala/com/cloudera/sparkts/models/ARIMA.scala
    https://github.com/robjhyndman/forecast/blob/master/R/arima.R
    https://github.com/tgsmith61591/pyramid
        ****/
    private ArimaAutoFit() {
    }

    // (0,0,0,0) is ignored for now
    private static final List<Quadruple<Integer, Integer, Integer, Integer>> initialpqPQValues = new ArrayList<Quadruple<Integer, Integer, Integer, Integer>>() {{
        add(Quadruple.of(2, 2, 1, 1));
        add(Quadruple.of(1, 0, 1, 0));
        add(Quadruple.of(0, 1, 0, 1));
    }};

    // autoFit returns Triplet of < ArimaModel, D, d >
    public static ArimaParams autoFit(double[] ts, int maxp, int maxd, int maxq, int maxP, int maxD, int maxQ, int m, boolean seasonal, int maxIterations) {

        // seasonality difference test
        int D = 0;
        if (seasonal && m > 1) {
            D = CanovaHansenTest.estimateSeasonalDifferencingTerm(ts, m, maxD);
            if (D > 0) {
                ts = dropFirstD(differencesOfOrderD(ts, D, m), D * m);
            }
        }

        // stationarity test
        double kpssSignificance = 0.05;
        // Our first task is to choose a differencing order.
        // Following R's forecast::auto.arima -> forecast::ndiffs -> tseries::kpss.test, we test for
        // level stationarity and return lowest differencing order that results in level stationarity.
        int dOpt = -1;
        for (int d = 0; d <= maxd; d++) {
            double[] diffTs = dropFirstD(differencesOfOrderD(ts, d, 1), d);
            double stat = KPSSTest.evaluate(diffTs);
            if (stat < KPSSTest.kpssConstantCriticalValues.get(kpssSignificance)) {
                dOpt = d;
                break;
            }
        }

        if (dOpt == -1) {
            throw new RuntimeException("stationarity not achieved with differencing order <= " + maxd);
        }
        double[] diffedTs = differencesOfOrderD(ts, dOpt, 1);

        // can't return model from here, D and d are not set in it.
        ArimaModel model = findBestARMAModel(diffedTs, maxp, maxq, maxP, maxQ, m, seasonal, maxIterations);
        if (model == null) {
            return new ArimaParams(0, dOpt, 0, 0, D, 0, m, true);
        }
        return new ArimaParams(model.getParams().p, dOpt, model.getParams().q, model.getParams().P, D, model.getParams().Q, m);
    }


    /**
     * Performs differencing of order `d`. This means we recursively difference a vector a total of
     * d-times. So that d = 2 is a vector of the differences of differences. Note that for each
     * difference level, d_i, the element at ts(d_i - 1) corresponds to the value in the prior
     * iteration.
     *
     * @param ts time series to difference
     * @param d  order of differencing
     * @return a array of the same length differenced to order d
     */
    public static double[] differencesOfOrderD(double[] ts, int d, int lag) {
        // we create 2 copies to avoid copying with every call, and simply swap them as necessary
        // for higher order differencing
        double[] diffedTs = ts.clone(), origTs = ts.clone();
        double[] swap = null;
        for (int i = 1; i <= d; i++) {
            swap = origTs;
            origTs = diffedTs;
            diffedTs = swap;
            differencesAtLag(origTs, diffedTs, lag, i);
        }

        return diffedTs;
    }

    public static double[] dropFirstD(double[] ts, int d) {
        return Arrays.copyOfRange(ts, d, ts.length);
    }

    /**
     * Difference a vector with respect to the m-th prior element. Size-preserving by leaving first
     * `m` elements intact. This is the inverse of the `inverseDifferences` function.
     *
     * @param ts         Series to difference
     * @param destTs     Series to store the differenced values (and return for convenience)
     * @param lag        The difference lag (e.g. x means destTs(i) = ts(i) - ts(i - x), etc)
     * @param startIndex the starting index for the differencing. Must be at least equal to lag
     * @return the differenced array, for convenience
     */
    private static double[] differencesAtLag(double[] ts, double[] destTs, int lag, int startIndex) {

        double[] diffedTs = destTs != null ? destTs : ts.clone();

        if (lag == 0) {
            return diffedTs;
        }
        for (int i = 0; i < diffedTs.length; i++) {
            // elements prior to starting point are copied over without modification
            diffedTs[i] = i < startIndex * lag ? ts[i] : ts[i] - ts[i - lag];
        }
        return diffedTs;
    }

    private static ArimaModel findBestARMAModel(double[] diffedTs, int maxp, int maxq, int maxP, int maxQ, int m, boolean seasonal, int maxIterations) {

        ArimaModel curBestModel = null;
        boolean done = false;
        double curBestAIC = Double.MAX_VALUE;
        int iterations = 0;

        List<Quadruple<Integer, Integer, Integer, Integer>> pastParams = new ArrayList<>();

        List<Quadruple<Integer, Integer, Integer, Integer>> nextParams = new ArrayList<>();
        for (Quadruple<Integer, Integer, Integer, Integer> entry : initialpqPQValues) {
            nextParams.add(Quadruple.of(entry.getV1(), entry.getV2(), entry.getV3(), entry.getV4()));
        }

        while (!done) {
            pastParams.addAll(nextParams);
            List<ArimaModel> models = new ArrayList<>();
            for (Quadruple<Integer, Integer, Integer, Integer> params : nextParams) {
                boolean includeConstant = params.getV1() == 0 && params.getV2() == 0 && params.getV3() == 0 && params.getV4() == 0;
                ArimaParams newArimaParams = new ArimaParams(params.getV1(), 0, params.getV2(), seasonal ? params.getV3() : 0, 0, seasonal ? params.getV4() : 0, m, includeConstant);
                try {
                    models.add(ArimaSolver.estimateARIMA(newArimaParams, diffedTs, diffedTs.length, diffedTs.length + 1));
                } catch (Exception e) {
                    // do nothing here
                }
                iterations++;
            }
            boolean updated = false;
            for (ArimaModel model : models) {
                double approxAIC = approxAIC(diffedTs, model.getParams());
                if (approxAIC < curBestAIC) {
                    curBestModel = model;
                    curBestAIC = approxAIC;
                    updated = true;
                }
            }
            int p;
            if (curBestModel == null) {
                // means no model fitted from initialpqPQValues
                return null;
            } else {
                p = curBestModel.getParams().p;
            }
            int q = curBestModel.getParams().q;
            int P = curBestModel.getParams().P;
            int Q = curBestModel.getParams().Q;
            if (!updated) {
                done = true;
                //break;
            } else if (!seasonal) {
                List<Quadruple<Integer, Integer, Integer, Integer>> surroundingParams = new ArrayList<>();
                Integer[] deltas = {-1, 0, 1};
                for (int pDelta : deltas) {
                    for (int qDelta : deltas) {
                        if (pDelta != 0 || qDelta != 0) {
                            surroundingParams.add(Quadruple.of(p + pDelta, q + qDelta, 0, 0));
                        }
                    }
                }
                nextParams = surroundingParams.stream().filter(params -> !pastParams.contains(params) && params.getV1() >= 0
                        && params.getV1() <= maxp && params.getV2() >= 0 && params.getV2() <= maxq).collect(Collectors.toList());

            } else {
                if (iterations > maxIterations) {
                    done = true;
                    continue;
                }
                List<Quadruple<Integer, Integer, Integer, Integer>> surroundingParams = new ArrayList<>();
                //P fluctuations:
                surroundingParams.add(Quadruple.of(p, q, P - 1, Q));
                surroundingParams.add(Quadruple.of(p, q, P + 1, Q));

                //Q fluctuations
                surroundingParams.add(Quadruple.of(p, q, P, Q - 1));
                surroundingParams.add(Quadruple.of(p, q, P, Q + 1));

                // P & Q fluctuations
                surroundingParams.add(Quadruple.of(p, q, P - 1, Q - 1));
                surroundingParams.add(Quadruple.of(p, q, P + 1, Q + 1));

                // p fluctuations
                surroundingParams.add(Quadruple.of(p - 1, q, P, Q));
                surroundingParams.add(Quadruple.of(p + 1, q, P, Q));

                // q fluctuations
                surroundingParams.add(Quadruple.of(p - 1, q, P, Q));
                surroundingParams.add(Quadruple.of(p + 1, q, P, Q));

                //p & q fluctuations:
                surroundingParams.add(Quadruple.of(p - 1, q - 1, P, Q));
                surroundingParams.add(Quadruple.of(p + 1, q + 1, P, Q));

                nextParams = surroundingParams.stream().filter(params -> !pastParams.contains(params) && params.getV1() >= 0
                        && params.getV1() <= maxp && params.getV2() >= 0 && params.getV2() <= maxq && params.getV3() >= 0 && params.getV3() <= maxP && params.getV4() >= 0 && params.getV4() <= maxQ).collect(Collectors.toList());

            }
        }
        return curBestModel;
    }


    /**
     * Calculates an approximation to the Akaike Information Criterion (AIC). This is an approximation
     * as we use the conditional likelihood, rather than the exact likelihood. Please see
     * [[https://en.wikipedia.org/wiki/Akaike_information_criterion]] for more information on this
     * measure.
     *
     * @param ts the timeseries to evaluate under current model
     * @return an approximation to the AIC under the current model
     */
    private static double approxAIC(double[] ts, ArimaParams params) {
        double conditionalLogLikelihood = logLikelihoodCSSARMA(ts, params);
        return -2 * conditionalLogLikelihood + 2 * (params.p + params.q + params.P + params.Q);
    }


    /**
     * log likelihood based on conditional sum of squares. In contrast to logLikelihoodCSS the array
     * provided should correspond to an already differenced array, so that the function below
     * corresponds to the log likelihood for the ARMA rather than the ARIMA process
     *
     * @param diffedY differenced array
     * @return log likelihood of ARMA
     */
    private static double logLikelihoodCSSARMA(double[] diffedY, ArimaParams params) {
        int n = diffedY.length;

        int _dp = params.getDegreeP();
        int _dq = params.getDegreeQ();
        int start_idx = (_dp > _dq) ? _dp : _dq;

        double[] yHat = new double[n];
        double[] yVec = diffedY.clone();

        iterateARMA(yVec, yHat, params, start_idx);
        int iterated_N = n - start_idx;
        // drop first maxLag terms, since we can't estimate residuals there, since no
        // AR(n) terms available
        double css = 0.0;
        for (int i = start_idx; i < n; i++) {
            css += Math.pow(yVec[i] - yHat[i], 2);
        }
        double sigma2 = css / iterated_N;
        return (-iterated_N / 2) * Math.log(2 * Math.PI * sigma2) - css / (2 * sigma2);
    }


    /**
     * Perform operations with the AR and MA terms, based on the time series `ts` and the errors
     * based off of `goldStandard` or `errors`, combined with elements from the series `dest`.
     * Weights for terms are taken from the current model configuration.
     * So for example: iterateARMA(series1, series_of_zeros,  _ + _ , goldStandard = series1,
     * initErrors = null)
     * calculates the 1-step ahead ARMA forecasts for series1 assuming current coefficients, and
     * initial MA errors of 0.
     *
     * @param ts   Time series to use for AR terms
     * @param dest Time series holding initial values at each index
     * @return the time series resulting from the interaction of the parameters with the model's
     * coefficients
     */
    private static double[] iterateARMA(double[] ts, double[] dest, ArimaParams params, int start_idx) {

        final int n = ts.length;
        final double[] errors = new double[n];

        // populate errors and forecasts
        for (int j = start_idx; j < n; ++j) {
            final double forecast = params.forecastOnePointARMA(ts, errors, j);
            final double error = ts[j] - forecast;
            errors[j] = error;
        }
        // now we can forecast
        for (int j = start_idx; j < n; ++j) {
            final double forecast = params.forecastOnePointARMA(ts, errors, j);
            dest[j] = forecast;
        }
        // return forecasted values
        return dest;


        /*
            SCALA code for non seasonality to iterate ARMA
            https://github.com/sryza/spark-timeseries/blob/master/src/main/scala/com/cloudera/sparkts/models/ARIMA.scala

        double[] maTerms = new double[params.q];
        int intercept = hasIntercept ? 1 : 0;
        // maximum lag
        int i = start_idx;
        int j;

        double error;

        while (i < n) {
            j = 0;
            // intercept
            //dest[i] += params.getCurrentARCoefficients()[0]
            // autoregressive terms
            while (j < params.p && i - j - 1 >= 0) {
                dest[i] += ts[i - j - 1] * params.getCurrentARCoefficients()[j + 1];
                j += 1;
            }
            // moving average terms
            j = 0;
            while (j < params.q) {
                dest[i] += maTerms[j] * params.getCurrentMACoefficients()[j + 1];
                j += 1;
            }

            error = goldStandard[i] - dest[i];
            updateMAErrors(maTerms, error);
            i += 1;
        }
        return dest;

        */
    }

/*

    */
/**
     * Updates the error vector in place with a new (more recent) error
     * The newest error is placed in position 0, while older errors "fall off the end"
     *
     * @param errs     array of errors of length q in ARIMA(p, d, q), holds errors for t-1 through t-q
     * @param newError the error at time t
     * @return a modified array with the latest error placed into index 0
     *//*

    private static double[] updateMAErrors(double[] errs, double newError) {
        int n = errs.length;
        int i = 0;
        while (i < n - 1) {
            errs[i + 1] = errs[i];
            i += 1;
        }
        if (n > 0) {
            errs[0] = newError;
        }
        return errs;
    }
*/


}
