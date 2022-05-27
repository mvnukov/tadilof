import com.github.chen0040.data.evaluators.BinaryClassifierEvaluator;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataQuery;
import com.github.chen0040.data.frame.Sampler;
import com.github.chen0040.lof.LOF;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

public class LofMain {

    private static Logger logger = LoggerFactory.getLogger(LofMain.class);

    private static Random random = new Random();

    public static double rand() {
        return random.nextDouble();
    }

    public static double rand(double lower, double upper) {
        return rand() * (upper - lower) + lower;
    }

    public static void main(String[] args) {
        DataQuery.DataFrameQueryBuilder schema = DataQuery.blank()
                .newInput("c1")
                .newInput("c2")
                .newOutput("anomaly")
                .end();

        Sampler.DataSampleBuilder negativeSampler = new Sampler()
                .forColumn("c1").generate((name, index) -> randn() * 0.3 + (index % 2 == 0 ? -2 : 2))
                .forColumn("c2").generate((name, index) -> randn() * 0.3 + (index % 2 == 0 ? -2 : 2))
                .forColumn("anomaly").generate((name, index) -> 0.0)
                .end();

        Sampler.DataSampleBuilder positiveSampler = new Sampler()
                .forColumn("c1").generate((name, index) -> rand(-4, 4))
                .forColumn("c2").generate((name, index) -> rand(-4, 4))
                .forColumn("anomaly").generate((name, index) -> 1.0)
                .end();

        DataFrame data = schema.build();

        data = negativeSampler.sample(data, 20);
        data = positiveSampler.sample(data, 20);

        logger.info(data.head(10));

        LOF method = new LOF();
        method.setParallel(true);
        method.setMinPtsLB(3);
        method.setMinPtsUB(10);
        method.setThreshold(0.5);
        DataFrame learnedData = method.fitAndTransform(data);

        BinaryClassifierEvaluator evaluator = new BinaryClassifierEvaluator();

        for (int i = 0; i < learnedData.rowCount(); ++i) {
            boolean predicted = learnedData.row(i).categoricalTarget().equals("1");
            boolean actual = data.row(i).target() == 1.0;
            evaluator.evaluate(actual, predicted);
            logger.info("predicted: {}\texpected: {}", predicted, actual);
        }

    }

    public static double randn() {
        double u1 = rand();
        double u2 = rand();
        double r = Math.sqrt(-2.0 * Math.log(u1));
        double theta = 2.0 * Math.PI * u2;
        return r * Math.sin(theta);
    }

}
