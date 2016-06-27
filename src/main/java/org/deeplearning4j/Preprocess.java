package org.deeplearning4j;

import org.canova.api.io.labels.ParentPathLabelGenerator;
import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.BalanceMinibatches;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.io.File;

/**
 * Hello world!
 *
 */
public class Preprocess {
    public static void main( String[] args ) throws Exception {
        int height = 128;
        int width = 128;
        int channels = 3;
        int batchSize = 64;

        ImageRecordReader reader = new ImageRecordReader(height,width,channels,new ParentPathLabelGenerator());
        reader.initialize(new FileSplit(new File("data")));
        DataSetIterator iterator = new RecordReaderDataSetIterator(reader,batchSize);
        DataNormalization scaler = new NormalizerStandardize();
        scaler.fit(iterator);
        iterator.reset();
        while(iterator.hasNext()) {
            DataSet next = iterator.next();
            scaler.transform(next);
            System.out.println(iterator.next().getFeatureMatrix().shapeInfoToString());
        }

        iterator.reset();
        BalanceMinibatches balanceMinibatches = BalanceMinibatches.builder()
                .dataSetIterator(iterator).miniBatchSize(batchSize).numLabels(new File("data").list().length)
                .rootDir(new File("minibatches")).rootSaveDir(new File("minibatchessave")).dataNormalization(scaler)
                .build();
        balanceMinibatches.balance();
        scaler.save(new File("mean.bin"),new File("std.bin"));

    }
}
