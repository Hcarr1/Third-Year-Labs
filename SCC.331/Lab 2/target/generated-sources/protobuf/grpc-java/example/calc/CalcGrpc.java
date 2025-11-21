package example.calc;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 */
@io.grpc.stub.annotations.GrpcGenerated
public final class CalcGrpc {

  private CalcGrpc() {}

  public static final java.lang.String SERVICE_NAME = "Calc";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<example.calc.FactRequest,
      example.calc.FactResponse> getFactorialMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "Factorial",
      requestType = example.calc.FactRequest.class,
      responseType = example.calc.FactResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<example.calc.FactRequest,
      example.calc.FactResponse> getFactorialMethod() {
    io.grpc.MethodDescriptor<example.calc.FactRequest, example.calc.FactResponse> getFactorialMethod;
    if ((getFactorialMethod = CalcGrpc.getFactorialMethod) == null) {
      synchronized (CalcGrpc.class) {
        if ((getFactorialMethod = CalcGrpc.getFactorialMethod) == null) {
          CalcGrpc.getFactorialMethod = getFactorialMethod =
              io.grpc.MethodDescriptor.<example.calc.FactRequest, example.calc.FactResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "Factorial"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  example.calc.FactRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  example.calc.FactResponse.getDefaultInstance()))
              .setSchemaDescriptor(new CalcMethodDescriptorSupplier("Factorial"))
              .build();
        }
      }
    }
    return getFactorialMethod;
  }

  private static volatile io.grpc.MethodDescriptor<example.calc.SumRequest,
      example.calc.SumResponse> getSumMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "Sum",
      requestType = example.calc.SumRequest.class,
      responseType = example.calc.SumResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<example.calc.SumRequest,
      example.calc.SumResponse> getSumMethod() {
    io.grpc.MethodDescriptor<example.calc.SumRequest, example.calc.SumResponse> getSumMethod;
    if ((getSumMethod = CalcGrpc.getSumMethod) == null) {
      synchronized (CalcGrpc.class) {
        if ((getSumMethod = CalcGrpc.getSumMethod) == null) {
          CalcGrpc.getSumMethod = getSumMethod =
              io.grpc.MethodDescriptor.<example.calc.SumRequest, example.calc.SumResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "Sum"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  example.calc.SumRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  example.calc.SumResponse.getDefaultInstance()))
              .setSchemaDescriptor(new CalcMethodDescriptorSupplier("Sum"))
              .build();
        }
      }
    }
    return getSumMethod;
  }

  private static volatile io.grpc.MethodDescriptor<example.calc.ProductRequest,
      example.calc.ProductResponse> getProductMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "Product",
      requestType = example.calc.ProductRequest.class,
      responseType = example.calc.ProductResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<example.calc.ProductRequest,
      example.calc.ProductResponse> getProductMethod() {
    io.grpc.MethodDescriptor<example.calc.ProductRequest, example.calc.ProductResponse> getProductMethod;
    if ((getProductMethod = CalcGrpc.getProductMethod) == null) {
      synchronized (CalcGrpc.class) {
        if ((getProductMethod = CalcGrpc.getProductMethod) == null) {
          CalcGrpc.getProductMethod = getProductMethod =
              io.grpc.MethodDescriptor.<example.calc.ProductRequest, example.calc.ProductResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "Product"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  example.calc.ProductRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  example.calc.ProductResponse.getDefaultInstance()))
              .setSchemaDescriptor(new CalcMethodDescriptorSupplier("Product"))
              .build();
        }
      }
    }
    return getProductMethod;
  }

  private static volatile io.grpc.MethodDescriptor<example.calc.SqrtRequest,
      example.calc.SqrtResponse> getSqrtMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "Sqrt",
      requestType = example.calc.SqrtRequest.class,
      responseType = example.calc.SqrtResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<example.calc.SqrtRequest,
      example.calc.SqrtResponse> getSqrtMethod() {
    io.grpc.MethodDescriptor<example.calc.SqrtRequest, example.calc.SqrtResponse> getSqrtMethod;
    if ((getSqrtMethod = CalcGrpc.getSqrtMethod) == null) {
      synchronized (CalcGrpc.class) {
        if ((getSqrtMethod = CalcGrpc.getSqrtMethod) == null) {
          CalcGrpc.getSqrtMethod = getSqrtMethod =
              io.grpc.MethodDescriptor.<example.calc.SqrtRequest, example.calc.SqrtResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "Sqrt"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  example.calc.SqrtRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  example.calc.SqrtResponse.getDefaultInstance()))
              .setSchemaDescriptor(new CalcMethodDescriptorSupplier("Sqrt"))
              .build();
        }
      }
    }
    return getSqrtMethod;
  }

  private static volatile io.grpc.MethodDescriptor<example.calc.FactorizeRequest,
      example.calc.FactorizeResponse> getFactorizeMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "Factorize",
      requestType = example.calc.FactorizeRequest.class,
      responseType = example.calc.FactorizeResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.SERVER_STREAMING)
  public static io.grpc.MethodDescriptor<example.calc.FactorizeRequest,
      example.calc.FactorizeResponse> getFactorizeMethod() {
    io.grpc.MethodDescriptor<example.calc.FactorizeRequest, example.calc.FactorizeResponse> getFactorizeMethod;
    if ((getFactorizeMethod = CalcGrpc.getFactorizeMethod) == null) {
      synchronized (CalcGrpc.class) {
        if ((getFactorizeMethod = CalcGrpc.getFactorizeMethod) == null) {
          CalcGrpc.getFactorizeMethod = getFactorizeMethod =
              io.grpc.MethodDescriptor.<example.calc.FactorizeRequest, example.calc.FactorizeResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.SERVER_STREAMING)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "Factorize"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  example.calc.FactorizeRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  example.calc.FactorizeResponse.getDefaultInstance()))
              .setSchemaDescriptor(new CalcMethodDescriptorSupplier("Factorize"))
              .build();
        }
      }
    }
    return getFactorizeMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static CalcStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<CalcStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<CalcStub>() {
        @java.lang.Override
        public CalcStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new CalcStub(channel, callOptions);
        }
      };
    return CalcStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports all types of calls on the service
   */
  public static CalcBlockingV2Stub newBlockingV2Stub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<CalcBlockingV2Stub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<CalcBlockingV2Stub>() {
        @java.lang.Override
        public CalcBlockingV2Stub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new CalcBlockingV2Stub(channel, callOptions);
        }
      };
    return CalcBlockingV2Stub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static CalcBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<CalcBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<CalcBlockingStub>() {
        @java.lang.Override
        public CalcBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new CalcBlockingStub(channel, callOptions);
        }
      };
    return CalcBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static CalcFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<CalcFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<CalcFutureStub>() {
        @java.lang.Override
        public CalcFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new CalcFutureStub(channel, callOptions);
        }
      };
    return CalcFutureStub.newStub(factory, channel);
  }

  /**
   */
  public interface AsyncService {

    /**
     */
    default void factorial(example.calc.FactRequest request,
        io.grpc.stub.StreamObserver<example.calc.FactResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getFactorialMethod(), responseObserver);
    }

    /**
     */
    default void sum(example.calc.SumRequest request,
        io.grpc.stub.StreamObserver<example.calc.SumResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getSumMethod(), responseObserver);
    }

    /**
     */
    default void product(example.calc.ProductRequest request,
        io.grpc.stub.StreamObserver<example.calc.ProductResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getProductMethod(), responseObserver);
    }

    /**
     */
    default void sqrt(example.calc.SqrtRequest request,
        io.grpc.stub.StreamObserver<example.calc.SqrtResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getSqrtMethod(), responseObserver);
    }

    /**
     */
    default void factorize(example.calc.FactorizeRequest request,
        io.grpc.stub.StreamObserver<example.calc.FactorizeResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getFactorizeMethod(), responseObserver);
    }
  }

  /**
   * Base class for the server implementation of the service Calc.
   */
  public static abstract class CalcImplBase
      implements io.grpc.BindableService, AsyncService {

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return CalcGrpc.bindService(this);
    }
  }

  /**
   * A stub to allow clients to do asynchronous rpc calls to service Calc.
   */
  public static final class CalcStub
      extends io.grpc.stub.AbstractAsyncStub<CalcStub> {
    private CalcStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected CalcStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new CalcStub(channel, callOptions);
    }

    /**
     */
    public void factorial(example.calc.FactRequest request,
        io.grpc.stub.StreamObserver<example.calc.FactResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getFactorialMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void sum(example.calc.SumRequest request,
        io.grpc.stub.StreamObserver<example.calc.SumResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getSumMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void product(example.calc.ProductRequest request,
        io.grpc.stub.StreamObserver<example.calc.ProductResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getProductMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void sqrt(example.calc.SqrtRequest request,
        io.grpc.stub.StreamObserver<example.calc.SqrtResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getSqrtMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void factorize(example.calc.FactorizeRequest request,
        io.grpc.stub.StreamObserver<example.calc.FactorizeResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncServerStreamingCall(
          getChannel().newCall(getFactorizeMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   * A stub to allow clients to do synchronous rpc calls to service Calc.
   */
  public static final class CalcBlockingV2Stub
      extends io.grpc.stub.AbstractBlockingStub<CalcBlockingV2Stub> {
    private CalcBlockingV2Stub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected CalcBlockingV2Stub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new CalcBlockingV2Stub(channel, callOptions);
    }

    /**
     */
    public example.calc.FactResponse factorial(example.calc.FactRequest request) throws io.grpc.StatusException {
      return io.grpc.stub.ClientCalls.blockingV2UnaryCall(
          getChannel(), getFactorialMethod(), getCallOptions(), request);
    }

    /**
     */
    public example.calc.SumResponse sum(example.calc.SumRequest request) throws io.grpc.StatusException {
      return io.grpc.stub.ClientCalls.blockingV2UnaryCall(
          getChannel(), getSumMethod(), getCallOptions(), request);
    }

    /**
     */
    public example.calc.ProductResponse product(example.calc.ProductRequest request) throws io.grpc.StatusException {
      return io.grpc.stub.ClientCalls.blockingV2UnaryCall(
          getChannel(), getProductMethod(), getCallOptions(), request);
    }

    /**
     */
    public example.calc.SqrtResponse sqrt(example.calc.SqrtRequest request) throws io.grpc.StatusException {
      return io.grpc.stub.ClientCalls.blockingV2UnaryCall(
          getChannel(), getSqrtMethod(), getCallOptions(), request);
    }

    /**
     */
    @io.grpc.ExperimentalApi("https://github.com/grpc/grpc-java/issues/10918")
    public io.grpc.stub.BlockingClientCall<?, example.calc.FactorizeResponse>
        factorize(example.calc.FactorizeRequest request) {
      return io.grpc.stub.ClientCalls.blockingV2ServerStreamingCall(
          getChannel(), getFactorizeMethod(), getCallOptions(), request);
    }
  }

  /**
   * A stub to allow clients to do limited synchronous rpc calls to service Calc.
   */
  public static final class CalcBlockingStub
      extends io.grpc.stub.AbstractBlockingStub<CalcBlockingStub> {
    private CalcBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected CalcBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new CalcBlockingStub(channel, callOptions);
    }

    /**
     */
    public example.calc.FactResponse factorial(example.calc.FactRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getFactorialMethod(), getCallOptions(), request);
    }

    /**
     */
    public example.calc.SumResponse sum(example.calc.SumRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getSumMethod(), getCallOptions(), request);
    }

    /**
     */
    public example.calc.ProductResponse product(example.calc.ProductRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getProductMethod(), getCallOptions(), request);
    }

    /**
     */
    public example.calc.SqrtResponse sqrt(example.calc.SqrtRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getSqrtMethod(), getCallOptions(), request);
    }

    /**
     */
    public java.util.Iterator<example.calc.FactorizeResponse> factorize(
        example.calc.FactorizeRequest request) {
      return io.grpc.stub.ClientCalls.blockingServerStreamingCall(
          getChannel(), getFactorizeMethod(), getCallOptions(), request);
    }
  }

  /**
   * A stub to allow clients to do ListenableFuture-style rpc calls to service Calc.
   */
  public static final class CalcFutureStub
      extends io.grpc.stub.AbstractFutureStub<CalcFutureStub> {
    private CalcFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected CalcFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new CalcFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<example.calc.FactResponse> factorial(
        example.calc.FactRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getFactorialMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<example.calc.SumResponse> sum(
        example.calc.SumRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getSumMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<example.calc.ProductResponse> product(
        example.calc.ProductRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getProductMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<example.calc.SqrtResponse> sqrt(
        example.calc.SqrtRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getSqrtMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_FACTORIAL = 0;
  private static final int METHODID_SUM = 1;
  private static final int METHODID_PRODUCT = 2;
  private static final int METHODID_SQRT = 3;
  private static final int METHODID_FACTORIZE = 4;

  private static final class MethodHandlers<Req, Resp> implements
      io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {
    private final AsyncService serviceImpl;
    private final int methodId;

    MethodHandlers(AsyncService serviceImpl, int methodId) {
      this.serviceImpl = serviceImpl;
      this.methodId = methodId;
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_FACTORIAL:
          serviceImpl.factorial((example.calc.FactRequest) request,
              (io.grpc.stub.StreamObserver<example.calc.FactResponse>) responseObserver);
          break;
        case METHODID_SUM:
          serviceImpl.sum((example.calc.SumRequest) request,
              (io.grpc.stub.StreamObserver<example.calc.SumResponse>) responseObserver);
          break;
        case METHODID_PRODUCT:
          serviceImpl.product((example.calc.ProductRequest) request,
              (io.grpc.stub.StreamObserver<example.calc.ProductResponse>) responseObserver);
          break;
        case METHODID_SQRT:
          serviceImpl.sqrt((example.calc.SqrtRequest) request,
              (io.grpc.stub.StreamObserver<example.calc.SqrtResponse>) responseObserver);
          break;
        case METHODID_FACTORIZE:
          serviceImpl.factorize((example.calc.FactorizeRequest) request,
              (io.grpc.stub.StreamObserver<example.calc.FactorizeResponse>) responseObserver);
          break;
        default:
          throw new AssertionError();
      }
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public io.grpc.stub.StreamObserver<Req> invoke(
        io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        default:
          throw new AssertionError();
      }
    }
  }

  public static final io.grpc.ServerServiceDefinition bindService(AsyncService service) {
    return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
        .addMethod(
          getFactorialMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              example.calc.FactRequest,
              example.calc.FactResponse>(
                service, METHODID_FACTORIAL)))
        .addMethod(
          getSumMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              example.calc.SumRequest,
              example.calc.SumResponse>(
                service, METHODID_SUM)))
        .addMethod(
          getProductMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              example.calc.ProductRequest,
              example.calc.ProductResponse>(
                service, METHODID_PRODUCT)))
        .addMethod(
          getSqrtMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              example.calc.SqrtRequest,
              example.calc.SqrtResponse>(
                service, METHODID_SQRT)))
        .addMethod(
          getFactorizeMethod(),
          io.grpc.stub.ServerCalls.asyncServerStreamingCall(
            new MethodHandlers<
              example.calc.FactorizeRequest,
              example.calc.FactorizeResponse>(
                service, METHODID_FACTORIZE)))
        .build();
  }

  private static abstract class CalcBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    CalcBaseDescriptorSupplier() {}

    @java.lang.Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return example.calc.CalcOuterClass.getDescriptor();
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("Calc");
    }
  }

  private static final class CalcFileDescriptorSupplier
      extends CalcBaseDescriptorSupplier {
    CalcFileDescriptorSupplier() {}
  }

  private static final class CalcMethodDescriptorSupplier
      extends CalcBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final java.lang.String methodName;

    CalcMethodDescriptorSupplier(java.lang.String methodName) {
      this.methodName = methodName;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.MethodDescriptor getMethodDescriptor() {
      return getServiceDescriptor().findMethodByName(methodName);
    }
  }

  private static volatile io.grpc.ServiceDescriptor serviceDescriptor;

  public static io.grpc.ServiceDescriptor getServiceDescriptor() {
    io.grpc.ServiceDescriptor result = serviceDescriptor;
    if (result == null) {
      synchronized (CalcGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new CalcFileDescriptorSupplier())
              .addMethod(getFactorialMethod())
              .addMethod(getSumMethod())
              .addMethod(getProductMethod())
              .addMethod(getSqrtMethod())
              .addMethod(getFactorizeMethod())
              .build();
        }
      }
    }
    return result;
  }
}
