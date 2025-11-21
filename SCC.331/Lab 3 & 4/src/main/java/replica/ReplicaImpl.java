package replica;

import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.rmi.server.UnicastRemoteObject;
import java.util.*;
import java.util.regex.Pattern;

public class ReplicaImpl extends UnicastRemoteObject
        implements PrimaryAPI, ReplicaControl {

    // in-memory key-value store
    private final Map<String,String> store = new HashMap<>();

    // am I currently the primary?
    private boolean isPrimary;

    // my ID for naming (e.g., "1" for name "replica1")
    private final String myId;

    // stubs to the other replicas that act as backups
    private final List<ReplicaControl> backups;

    // pattern to recognize replica bindings in the registry
    private static final Pattern REPLICA_NAME = Pattern.compile("^replica\\d+$");

    protected ReplicaImpl(String myId, boolean startAsPrimary, List<ReplicaControl> backupsList) throws RemoteException {

        super();
        this.myId = Objects.requireNonNull(myId, "myId");
        this.isPrimary = startAsPrimary;
        this.backups = new ArrayList<>(backupsList);
    }

    /**
     * ========== PrimaryAPI ==========
     */

    @Override
    public synchronized boolean handleClientPut(String key, String value) throws RemoteException {
        
        //TODO:
        // 1) Check if I am currently the primary. If not, print an error message and return false

        if (!isPrimary) {
            System.err.println("[Replica " + myId + "] ERROR: Received handleClientPut but I am not the primary!");
            return false;
        }

        // 2) Update local state (store) by adding the key, value pair

        this.store.put(key, value);

        // 3) Push full state to backups

        //????????????

        // 4) Return true (success), if all goes well.

        return true;
    }

    @Override
    public synchronized String handleClientGet(String key) throws RemoteException {
        //TODO:
        // 1) Retrieve the value corresponding to the key

        String value = this.store.get(key);

        // 2) Return null if the key does not exist; otherwise, return value

        if (value != null) {
            return value;
        }
        
        return null;
    }

    @Override
    public synchronized void pushFullState(Map<String,String> newState) throws RemoteException {

        //TODO:
        // Replace the store with the newState

        this.store = new HashMap<>(newState);

        System.out.println("[Replica " + myId + "] pushFullState applied. Store now: " + this.store);
    }
    @Override
    public synchronized void promoteToPrimary() throws RemoteException {
        System.out.println("[Replica " + myId + "] promoteToPrimary() called. Switching to primary.");
        this.isPrimary = true;

        try {
            List<ReplicaControl> discovered = discoverBackups();
            setBackups(discovered);
            System.out.println("[Replica " + myId + "] Discovery complete. Backups=" + discovered.size());
        } catch (Exception e) {
            System.err.println("[Replica " + myId + "] Discovery failed: " + e);
        }
    }

    @Override
    public boolean ping() throws RemoteException {
        return true;
    }

    // ========== Helpers ==========

    public synchronized void setBackups(List<ReplicaControl> newBackups) {
        this.backups.clear();
        this.backups.addAll(newBackups);
    }

    private List<ReplicaControl> discoverBackups() {
        List<ReplicaControl> discovered = new ArrayList<>();

        try {
            Registry reg = LocateRegistry.getRegistry(); 
            String[] names = reg.list();
            String myName = "replica" + myId;

            for (String name : names) {
                if (!REPLICA_NAME.matcher(name).matches()) 
                    continue;
                if (name.equals(myName)) 
                    continue; // skip self

                try {
                    Object obj = reg.lookup(name);
                    if (obj instanceof ReplicaControl) {
                        ReplicaControl stub = (ReplicaControl) obj;
                        if (stub.ping()) {
                            discovered.add(stub);
                            System.out.println("[Replica " + myId + "] Found live backup: " + name);
                        }
                    }
                } catch (RemoteException e) {
                    // Ignore stale or unreachable entries
                }
            }
        } catch (Exception e) {
            System.err.println("[Replica " + myId + "] Error listing registry: " + e);
        }

        return discovered;
    }
}
