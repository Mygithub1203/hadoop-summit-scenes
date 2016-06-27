package org.deeplearning4j;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.input.PortableDataStream;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.StandardScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.InputStream;

/**
 * Created by agibsonccc on 6/25/16.
 */
public class Train {
    public static void main(String[] args) throws Exception {
        DataSetIterator existingIterator = new ExistingMiniBatchDataSetIterator(new File("minibatchessave"));
        int channels = -1;
        int rows =  -1;
        int cols = -1;
        String activation = "leakyrelu";
        int iterations = 10;
        int seed = 123;
        OptimizationAlgorithm optimizer = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
        int numLabels = 6;
        WeightInit weightInit = WeightInit.XAVIER;
        MultiLayerNetwork multiLayerNetwork = null;
        SparkConf sparkConf = new SparkConf().setMaster("local[*]")
                .set("spark.driver.maxResultSize","3g")
                .setAppName("computer vision");
        SparkContext sparkContext = new SparkContext(sparkConf);
        SparkDl4jMultiLayer sparkDl4jMultiLayer = null;

        JavaRDD<DataSet> dataSetJavaRDD = sparkContext.binaryFiles("file:///home/agibsonccc/code/scene-classification/minibatchessave",8).toJavaRDD().map(v1 -> {
            PortableDataStream stream = v1._2();
            InputStream is = stream.open();
            DataSet d = new DataSet();
            d.load(is);
            stream.close();
            return d;
        });


        for(int i = 0; i < iterations; i++) {
            System.out.println("Iteration " + i);
            int batch = 0;
            //   while(existingIterator.hasNext()) {
            System.out.println("Processing batch " + batch);
            DataSet data = existingIterator.next();
            if(channels < 0) {
                channels = data.getFeatureMatrix().size(1);
                rows = data.getFeatureMatrix().size(2);
                cols = data.getFeatureMatrix().size(3);

                MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                        .weightInit(weightInit)
                        .seed(seed)
                        .activation(activation)
                        .iterations(1)
                        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                        .optimizationAlgo(optimizer)
                        .updater(Updater.NESTEROVS)
                        .learningRate(0.01)
                        .momentum(0.9)
                        .regularization(true)
                        .l2(0.04)
                        .list()
                        .layer(0, new ConvolutionLayer.Builder(5, 5)
                                .name("cnn1")
                                .nIn(channels)
                                .stride(1, 1)
                                .padding(2, 2)
                                .nOut(32)
                                .build())
                        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(3, 3)
                                .name("pool1")
                                .build())
                        .layer(2, new LocalResponseNormalization.Builder(3, 5e-05, 0.75).build())
                        .layer(3, new ConvolutionLayer.Builder(5, 5)
                                .name("cnn2")
                                .stride(1, 1)
                                .padding(2, 2)
                                .nOut(32)
                                .build())
                        .layer(4, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(3, 3)
                                .name("pool2")
                                .build())
                        .layer(5, new LocalResponseNormalization.Builder(3, 5e-05, 0.75).build())
                        .layer(6, new ConvolutionLayer.Builder(5, 5)
                                .name("cnn3")
                                .stride(1, 1)
                                .padding(2, 2)
                                .nOut(64)
                                .build())
                        .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(3, 3)
                                .name("pool3")
                                .build())
                        .layer(8, new DenseLayer.Builder()
                                .name("ffn1")
                                .nOut(250)
                                .dropOut(0.5)
                                .build())
                        .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nOut(numLabels)
                                .activation("softmax")
                                .build())
                        .backprop(true).pretrain(false)
                        .cnnInputSize(rows, cols, channels);
                multiLayerNetwork = new MultiLayerNetwork(builder.build());
                multiLayerNetwork.init();
                multiLayerNetwork.setListeners(new ScoreIterationListener(1));
                sparkDl4jMultiLayer = new SparkDl4jMultiLayer(sparkContext,multiLayerNetwork,new ParameterAveragingTrainingMaster(true,8,32,5,10,true));

            }




            sparkDl4jMultiLayer.fit(dataSetJavaRDD);
           /* System.out.println("Batch " + batch + " processed");
            batch++;*/
        }

        //  existingIterator.reset();
        //}


        ModelSerializer.writeModel(multiLayerNetwork,new File("model.zip"),false);
        System.out.println("Done training");
    }

}
