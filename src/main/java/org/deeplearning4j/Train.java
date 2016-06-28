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
        int iterations = args.length >= 1 ? Integer.parseInt(args[0]) : 10;
        int seed = 123;

        OptimizationAlgorithm optimizer = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
        int numLabels = 4;
        WeightInit weightInit = WeightInit.RELU;
        MultiLayerNetwork multiLayerNetwork = null;
        SparkConf sparkConf = new SparkConf().setMaster("local[*]")
                .set("spark.driver.maxResultSize","6g")
                .setAppName("computer vision");
        SparkContext sparkContext = new SparkContext(sparkConf);
        SparkDl4jMultiLayer sparkDl4jMultiLayer = null;
        int numWorkers = Runtime.getRuntime().availableProcessors();

        String sceneDirectory = System.getProperty("user.dir");
        JavaRDD<DataSet> dataSetJavaRDD = sparkContext.binaryFiles(String.format("file://%s/minibatchessave",sceneDirectory),numWorkers).toJavaRDD().map(v1 -> {
            PortableDataStream stream = v1._2();
            InputStream is = stream.open();
            DataSet d = new DataSet();
            d.load(is);
            stream.close();
            return d;
        });



        ParameterAveragingTrainingMaster trainingMaster = new ParameterAveragingTrainingMaster(true,numWorkers,32,1,10,true);

        if(new File("model-load.zip").exists()) {
            multiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork(new File("model-load.zip"));
            channels = 1;
            multiLayerNetwork.setListeners(new ScoreIterationListener(1));
            sparkDl4jMultiLayer = new SparkDl4jMultiLayer(sparkContext,multiLayerNetwork,trainingMaster);

        }


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
                        .iterations(3).rmsDecay(0.3)
                        .regularization(true)
                        .l1(1e-1)
                        .l2(2e-4)
                        .optimizationAlgo(optimizer)
                        .updater(Updater.RMSPROP)
                        .learningRate(0.01)
                        .momentum(0.5)
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
                sparkDl4jMultiLayer = new SparkDl4jMultiLayer(sparkContext,multiLayerNetwork,trainingMaster);

            }




            multiLayerNetwork =  sparkDl4jMultiLayer.fit(dataSetJavaRDD);
            ModelSerializer.writeModel(multiLayerNetwork,new File(String.format("model-%d.zip",i)),false);
            sparkDl4jMultiLayer = new SparkDl4jMultiLayer(sparkContext,multiLayerNetwork,trainingMaster);

           /* System.out.println("Batch " + batch + " processed");
            batch++;*/
        }

        //  existingIterator.reset();
        //}


        System.out.println("Done training");
    }

}
