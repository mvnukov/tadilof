package knn.benchmarks;

import org.openjdk.jmh.results.format.ResultFormatType;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.openjdk.jmh.runner.options.VerboseMode;

import java.util.concurrent.TimeUnit;

public class RunBenchmarks {

    public static void main(String[] args) throws RunnerException {
        final Options opt =
                new OptionsBuilder()
                        .include(KnnBenchmark.class.getSimpleName())
                        .include(IndexBenchmark.class.getSimpleName())
                        .mode(org.openjdk.jmh.annotations.Mode.All)
                        .timeUnit(TimeUnit.SECONDS)
                        .addProfiler("gc")
                        .addProfiler("stack")
                        .addProfiler("cl")
                        .addProfiler("jfr")
                        .resultFormat(ResultFormatType.JSON)
                        .forks(5)
                        .verbosity(VerboseMode.EXTRA)
                        .build();

        new org.openjdk.jmh.runner.Runner(opt).run();
    }
}
