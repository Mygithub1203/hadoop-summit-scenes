package org.deeplearning4j;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;

/**
 * Created by agibsonccc on 6/26/16.
 */
public class Predict {

    public static void main(String[] args) throws Exception {
        DataSet d;
        DataSetIterator existingIterator = new ExistingMiniBatchDataSetIterator(new File("minibatchessave"));
        Evaluation eval = new Evaluation(4);
        MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork(new File(args[0]));
        while(existingIterator.hasNext()) {
            d = existingIterator.next();
            INDArray predict = network.output(d.getFeatureMatrix());
            eval.eval(d.getLabels(),predict);
            System.out.println(eval.stats());

        }

        System.out.println(eval.stats());
    }

}
