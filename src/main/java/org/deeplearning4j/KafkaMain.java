package org.deeplearning4j;

import org.apache.camel.CamelContext;

import org.apache.camel.impl.DefaultCamelContext;
import org.deeplearning4j.streaming.routes.DL4jServeRouteBuilder;

/**
 * Created by agibsonccc on 6/28/16.
 */
public class KafkaMain {
    public static void main(String[] args) throws Exception  {
        CamelContext context = new DefaultCamelContext();
        String brokerList = "";
        String topicName = "";
        String outputPath = "";
        boolean computationGraph = false;
        int zooKeeperPort = 2181;
        int imageHeight = 128;
        int imageWidth = 128;
        int channels = 3;
        String camelOutputUri = "";
        context.addRoutes(DL4jServeRouteBuilder.builder()
                .computationGraph(computationGraph).zooKeeperPort(zooKeeperPort)
                .kafkaBroker(brokerList).consumingTopic(topicName).beforeProcessor(new ImageProcessor(imageHeight,imageWidth,channels))
                .modelUri(outputPath).outputUri(camelOutputUri).finalProcessor(exchange -> {
                            exchange.getIn().setBody(exchange.getIn().getBody().toString());
                            System.out.println(exchange.getIn().getBody().toString());
                        }
                ).build());
        context.startAllRoutes();

    }
}
