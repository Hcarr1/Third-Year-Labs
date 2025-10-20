import java.rmi.RemoteException;
import java.rmi.registry.Registry;
import java.rmi.server.UnicastRemoteObject;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

public class NodeImpl extends UnicastRemoteObject implements INode {

    private final String selfId;     // e.g., "node:3"
    private final String groupName;  // the group this process belongs to

    // Keep a handle to the registry so multicasting can do per-recipient lookups inside the impl
    private final Registry registry;

    // Map: group -> concurrent set of memberIds ("node:<id>")
    private final ConcurrentHashMap<String, Set<String>> groupMembers = new ConcurrentHashMap<>();

    public NodeImpl(String selfId, String groupName, Registry registry) throws RemoteException {
        super(0); // anonymous port
        this.selfId = selfId;
        this.groupName = groupName;
        this.registry = registry;

        // initialise the state
        groupMembers.putIfAbsent(groupName, ConcurrentHashMap.newKeySet());
        groupMembers.get(groupName).add(selfId); // include self
    }

    @Override
    public void joinGroup(String group, String fromId) throws RemoteException {
        // TODO:
        // - Add 'fromId' to the members.
        groupMembers.get(group).add(fromId);
        // - Optionally print a log only when a NEW member is added.
        System.out.println(fromId + " Joined group: " + group);
    }

    @Override
    public void leaveGroup(String group, String fromId) throws RemoteException {
        // TODO:
        // - Remove 'fromId' from the set if present.
        groupMembers.get(group).remove(fromId);
        // - Optionally print a log only when an EXISTING member actually gets removed.
        System.out.println(fromId + " Left group: " + group);
    }

    @Override
    public void sendToMember(String group, String fromId, String message) throws RemoteException {
        // TODO: simply print a message such as:
        System.out.printf("[%s] to=%s from=%s :: %s%n", group, selfId, fromId, message);
    }

    @Override
    public Set<String> getMembers(String group) throws RemoteException {
        // TODO
        return groupMembers.get(group);
    }

    public void multicast(String message) {
        Set<String> snapshot = new HashSet<>(localMembers());
        for (String memberId : snapshot) {
            if (memberId.equals(selfId)) continue;
            try {
                INode peer = (INode) registry.lookup(memberId);
                peer.sendToMember(groupName, selfId, message);
            } catch (Exception ignored) {
                // handle errors (see question 3)
            }
        }
    }

    public Set<String> localMembers() {
        return groupMembers.getOrDefault(groupName, java.util.Set.of());
    }

    public String selfId() { return selfId; }
    public String groupName() { return groupName; }
}