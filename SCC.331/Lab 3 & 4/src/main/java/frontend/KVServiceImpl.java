package frontend;

import io.grpc.stub.StreamObserver;
import kv.KVServiceGrpc;
import kv.PutRequest;
import kv.PutReply;
import kv.GetRequest;
import kv.GetReply;
import replica.PrimaryAPI;
import replica.ReplicaControl;

import java.rmi.RemoteException;
import java.util.ArrayList;
import java.util.List;

public class KVServiceImpl extends KVServiceGrpc.KVServiceImplBase {

    private volatile PrimaryAPI primaryStub;
    private final List<ReplicaControl> remainingBackups; // in promotion order

    public KVServiceImpl(PrimaryAPI initialPrimary, List<ReplicaControl> backupsInOrder) {
        this.primaryStub = initialPrimary;
        this.remainingBackups = new ArrayList<>(backupsInOrder);
    }

    /**
     * Helper to promote backup if primary died.
     */
    private synchronized void failoverToBackup(ReplicaControl backupStub) throws RemoteException {
        System.err.println("[FrontEnd] Primary appears dead. Promoting backup...");
        backupStub.promoteToPrimary();

        this.primaryStub = (PrimaryAPI) backupStub;
        System.err.println("[FrontEnd] Failover complete. Backup is now primary.");
    }

    @Override
    public void put(PutRequest request, StreamObserver<PutReply> responseObserver) {
        boolean ok = false;

        // TODO: 
        // 1) call handleClientPut on the primary replica (primaryStub)
        // 2) update the boolean ok (if needed)
        // 3) Implement a failure detector –
        //      a) Use try, catch (using appropriate exception) to detect failure
        //      b) If primary has failed, then failover to backup, using failoverToBackup()

        try {
            ok = this.primaryStub.handleClientPut(request.getKey(), request.getValue());
        } catch (RemoteException e) {
            if (!remainingBackups.isEmpty()) {
                ReplicaControl backupStub = remainingBackups.remove(0);
                try {
                    failoverToBackup(backupStub);
                    ok = this.primaryStub.handleClientPut(request.getKey(), request.getValue());
                } catch (RemoteException ex) {
                    System.err.println("[FrontEnd] Failover failed: " + ex.getMessage());
                    ok = false;
                }
            } else {
                System.err.println("[FrontEnd] No backups available for failover.");
                ok = false;
            }
        }

        // TODO: 
        // Build a reply and send it back to the client

        PutReply reply = PutReply.newBuilder().setSuccess(ok).build();
        responseObserver.onNext(reply);
        responseObserver.onCompleted();
    }

    @Override
    public void get(GetRequest request, StreamObserver<GetReply> responseObserver) {
        String value = null;
        boolean found = false;

        // TODO: 
        // 1) Call handleClientGet on the primary replica (primaryStub)
        // 2) update value and found variables (if needed)
        // 3) Implement a failure detector –
        //      a) Use try, catch (using appropriate exception) to detect failure
        //      b) If primary has failed, then failover to backup, using failoverToBackup()

        try {
            value = this.primaryStub.handleClientGet(request.getKey());
            if (value != null) {
                found = true;
            }
        } catch (RemoteException e) {
            if (!remainingBackups.isEmpty()) {
                ReplicaControl backupStub = remainingBackups.remove(0);
                try {
                    failoverToBackup(backupStub);
                    value = this.primaryStub.handleClientGet(request.getKey());
                    if (value != null) {
                        found = true;
                    }
                } catch (RemoteException ex) {
                    System.err.println("[FrontEnd] Failover failed: " + ex.getMessage());
                    found = false;
                }
            } else {
                System.err.println("[FrontEnd] No backups available for failover.");
                found = false;
            }
        }

        // TODO: 
        // Build a reply and send it back to the client

        GetReply reply = GetReply.newBuilder().setFound(found).setValue(value != null ? value : "").build();
        responseObserver.onNext(reply);
        responseObserver.onCompleted();
    }
}

