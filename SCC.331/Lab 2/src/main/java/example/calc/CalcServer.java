package example.calc;

import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;

import java.util.*;

public class CalcServer {
  public static void main(String[] args) throws Exception {
    int port = 50051;
    Server server = ServerBuilder.forPort(port)
            .addService(new CalcService())
            .build()
            .start();

    System.out.println("gRPC Calculator server running on port " + port);
    server.awaitTermination();
  }

  static class CalcService extends CalcGrpc.CalcImplBase {

    @Override
    public void factorial(FactRequest req, StreamObserver<FactResponse> out) {
      int n = req.getN();
      if (n < 0) {
        out.onError(Status.INVALID_ARGUMENT.withDescription("n must be >= 0").asRuntimeException());
        return;
      }
      // simple iterative factorial in long; guard a bit to avoid obvious overflow
      if (n > 20) {
        out.onError(Status.INVALID_ARGUMENT.withDescription("n too large (may overflow)").asRuntimeException());
        return;
      }
      long acc = 1L;
      for (int i = 2; i <= n; i++)
        acc *= i;
      out.onNext(FactResponse.newBuilder().setResult(acc).build());
      out.onCompleted();
    }

    @Override
    public void sum(SumRequest req, StreamObserver<SumResponse> out) {
      double r = req.getA() + req.getB();
      out.onNext(SumResponse.newBuilder().setResult(r).build());
      out.onCompleted();
    }


    @Override
    public void product(ProductRequest req, StreamObserver<ProductResponse> out) {
      double r = req.getA() * req.getB();
      out.onNext(ProductResponse.newBuilder().setResult(r).build());
      out.onCompleted();
    }

    @Override
    public void sqrt(SqrtRequest req, StreamObserver<SqrtResponse> out) {
      double n = req.getX();
      if (n <= 0) {
        out.onError(Status.INVALID_ARGUMENT.withDescription("n must be >= 0").asRuntimeException());
        return;
      }
      double r = Math.sqrt(n);
      out.onNext(SqrtResponse.newBuilder().setResult(r).build());
      out.onCompleted();
    }

    @Override
    public void factorize(FactorizeRequest req, StreamObserver<FactorizeResponse> out) {
      int n = req.getN();
      if (n < 2) {
        out.onError(Status.INVALID_ARGUMENT.withDescription("n must be > 1").asRuntimeException());
        return;
      }

      for (int i = 2; i <= n; i++) {
        while (n % i == 0) {
          System.out.println("factor: " + i);
          out.onNext(FactorizeResponse.newBuilder().setFactor(i).build());
          n /= i;
        }
      }
      out.onCompleted();
    }
  }
}