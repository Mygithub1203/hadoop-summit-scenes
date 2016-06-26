package org.deeplearning4j;

import org.canova.api.io.labels.ParentPathLabelGenerator;
import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.BalanceMinibatches;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.StandardScaler;

import java.io.File;

/**
 * Hello world!
 *
 */
public class Preprocess {
    public static void main( String[] args ) throws Exception {
        int height = 256;
        int width = 256;
        int channels = 3;
        int batchSize = 100;

        ImageRecordReader reader = new ImageRecordReader(height,width,channels,new ParentPathLabelGenerator());
        reader.initialize(new FileSplit(new File("data")));
        DataSetIterator iterator = new RecordReaderDataSetIterator(reader,batchSize);
        StandardScaler scaler = new StandardScaler();
        scaler.fit(iterator);
        iterator.reset();
        BalanceMinibatches balanceMinibatches = BalanceMinibatches.builder()
                .dataSetIterator(iterator).miniBatchSize(100).numLabels(new File("data").list().length)
                .rootDir(new File("minibatches")).rootSaveDir(new File("minibatchessave"))
                .build();
        balanceMinibatches.balance();
        scaler.save(new File("mean.bin"),new File("std.bin"));

    }
}
