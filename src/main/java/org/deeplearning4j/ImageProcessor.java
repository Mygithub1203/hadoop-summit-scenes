package org.deeplearning4j;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.apache.camel.Exchange;
import org.apache.camel.Processor;
import org.canova.api.split.InputStreamInputSplit;
import org.canova.image.loader.NativeImageLoader;
import org.canova.image.recordreader.ImageRecordReader;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.ByteArrayInputStream;

/**
 * Created by agibsonccc on 6/28/16.
 */
@Data
@Builder
@AllArgsConstructor
public class ImageProcessor implements Processor {
    private int height;
    private int width ;
    private int channels;

    /**
     * Processes the message exchange
     *
     * @param exchange the message exchange
     * @throws Exception if an internal processing error has occurred.
     */
    @Override
    public void process(Exchange exchange) throws Exception {
        byte[] body = (byte[]) exchange.getIn().getBody();
        NativeImageLoader imageLoader = new NativeImageLoader(height,width,channels);
        ByteArrayInputStream bis = new ByteArrayInputStream(body);
        INDArray arr = imageLoader.asMatrix(bis);
        bis.close();
        exchange.getIn().setBody(arr);

    }
}
