package org.deeplearning4j;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.StandardScaler;

import java.io.File;

/**
 * Created by agibsonccc on 6/25/16.
 */
public class Train {
    public static void main(String[] args) throws Exception {
        DataSetIterator existingIterator = new ExistingMiniBatchDataSetIterator(new File("minibatchessave"));
        StandardScaler standardScaler = new StandardScaler();
        standardScaler.load(new File("mean.bin"),new File("std.bin"));
        while(existingIterator.hasNext()) {
            DataSet data = existingIterator.next();
            standardScaler.transform(data);
        }
    }

}
