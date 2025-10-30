package example.calc;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;

public class CalcClient {
  public static void main(String[] args) {
    String host = "localhost";
    int port = 50051;

    ManagedChannel ch = ManagedChannelBuilder.forAddress(host, port)
        .usePlaintext()
        .build();

    try {
      var stub = CalcGrpc.newBlockingStub(ch);

      var sum = stub.sum(SumRequest.newBuilder().setA(1.5).setB(2.25).build());
      System.out.println("sum(1.5, 2.25) = " + sum.getResult());

      var fact = stub.factorial(FactRequest.newBuilder().setN(7).build());
      System.out.println("factorial(7) = " + fact.getResult());

      var product = stub.product(ProductRequest.newBuilder().setA(3).setB(4).build());
      System.out.println("product(3, 4) = " + product.getResult());

      var it = stub.factorize(FactorizeRequest.newBuilder().setN(84).build());
      System.out.print("factorize(84): ");
      while (it.hasNext()) {
        System.out.print(" " + it.next().getFactor());
      }

    } catch (StatusRuntimeException e) {
      System.err.println("RPC failed: " + e.getStatus());
    } finally {
      ch.shutdownNow();
    }
  }
}

