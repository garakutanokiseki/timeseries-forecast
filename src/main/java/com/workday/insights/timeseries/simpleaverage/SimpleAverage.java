package com.workday.insights.timeseries.simpleaverage;

import com.workday.insights.timeseries.arima.struct.ForecastResult;

import java.util.Arrays;

public final class SimpleAverage {
    private SimpleAverage(){
    }// pure static class

    public static ForecastResult forecast_simple_average(final double[] data, final int forecastSize) {

        try {
            ForecastResult forecastResult = new ForecastResult(new double [0], 0);

            if (data.length > 0) {
                double total = 0;
                for (double element : data) {
                    total += element;
                }
                double average = total / data.length;

                double resultForecast[] = new double[forecastSize];
                Arrays.fill(resultForecast, average);

                //Todo: setting DataVariance to Zero
                forecastResult = new ForecastResult(resultForecast, 0);
                // add logging messages
                forecastResult.log("forecast_simple_average");

            }
            return forecastResult;
        } catch (final Exception ex) {
            // failed to build ARIMA model
            throw new RuntimeException("Failed to build Simple Average forecast: " + ex.getMessage());
        }
    }
}
