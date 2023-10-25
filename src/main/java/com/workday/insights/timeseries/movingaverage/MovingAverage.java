package com.workday.insights.timeseries.movingaverage;

import com.sun.tools.javac.util.ArrayUtils;
import com.workday.insights.timeseries.arima.struct.ForecastResult;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

public final class MovingAverage {
    private MovingAverage(){

    }


    public static ForecastResult forecast_moving_average(final double[] data, final int forecastSize, final int windowSize) {

        try {
            ForecastResult forecastResult = new ForecastResult(new double [0], 0);

            ArrayList<Double> resultForecast = DoubleStream.of(data).boxed().collect(Collectors.toCollection(ArrayList::new));
            int skipCount = (windowSize==0 || windowSize>resultForecast.size())? 0 : (resultForecast.size() - windowSize);
            for (int i =0; i<forecastSize; i++){
                resultForecast.add(resultForecast.stream()
                        .skip(skipCount++)
                        .mapToDouble(Double::doubleValue)
                        .average()
                        .getAsDouble());

            }

            double[] res = new double[forecastSize];
            for (int i=0; i<forecastSize; i++){
                res[i] = resultForecast.get(data.length + i).doubleValue();
            }

            return new ForecastResult(res, 0);
        } catch (final Exception ex) {
            throw new RuntimeException("Failed to build Moving Average forecast: " + ex.getMessage());
        }

    }

}
