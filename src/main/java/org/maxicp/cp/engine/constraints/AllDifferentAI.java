package org.maxicp.cp.engine.constraints;

import org.json.JSONArray;
import org.json.JSONObject;
import org.maxicp.RunXCSP3.GlobalTimers;
import org.maxicp.RunXCSP3.SocketManager;
import org.maxicp.cp.engine.core.AbstractCPConstraint;
import org.maxicp.cp.engine.core.CPIntVar;
import org.maxicp.state.StateInt;

import java.io.*;

public class AllDifferentAI extends AbstractCPConstraint {

    private CPIntVar[] x;
    private int[] nonfixed; // keep the indexes of the unfixed vars
    private StateInt nbNonFixed;  // number of unfixed vars
    public static int nbCallPropagate = 0;  // count the number of calls to propagate

    public AllDifferentAI(CPIntVar... x) {
        super(x[0].getSolver());
        this.x = x;

        nonfixed = new int[x.length];
        for (int i = 0; i < nonfixed.length; i++) {nonfixed[i] = i;}
        nbNonFixed = this.getSolver().getStateManager().makeStateInt(x.length);
    }

    @Override
    public void post() {
        int s = nbNonFixed.value();
        for (int k = s - 1; k >= 0 ; k--) {
            int idx = nonfixed[k];
            if (!x[idx].isFixed()) {
                x[idx].propagateOnDomainChange(this);
            } else {
                // Swap
                s--;
                nonfixed[k] = nonfixed[s];
                nonfixed[s] = idx;
            }
        }
        nbNonFixed.setValue(s);

        // Inconsistency Exception can be triggered in some case
        checker();

    }

    @Override
    public void propagate() {

        nbCallPropagate++;

        // remove fixed values from other domains
        checker();

        // Build the Graph representation
        JSONObject graph = new JSONObject();
        for (int i = 0; i < x.length; i++) {
            // only keep non-fixed variables
            if (x[i].isFixed()) {
                continue;
            }

            int[] dom = new int[x[i].max() - x[i].min() + 1];
            int size = x[i].fillArray(dom);
            JSONArray values = new JSONArray();
            for (int j = 0; j < size; j++) {
                values.put(dom[j]);
            }
            graph.put(String.valueOf(i), values);
        }

        // if all variables are fixed, nothing to do
        if (graph.length() == 0) {
            return;
        }

        // Send the domain to the AI model and get the response
        try {

            GlobalTimers.allJavaTimer.pause();  // pause timer during communication
            JSONObject response = sendJsonAndGetResponse(graph);
            GlobalTimers.allJavaTimer.start();

            // Remove values from the domains based on the response
            for (String varName : response.keySet()) {
                int varIndex = Integer.parseInt(varName);
                JSONArray valsToRemove = response.getJSONArray(varName);
                for (int j = 0; j < valsToRemove.length(); j++) {
                    x[varIndex].remove(valsToRemove.getInt(j));
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
            SocketManager.getInstance().closeQuietly();
        }

    }

    public void checker() {
        int s = nbNonFixed.value();
        for (int k = s - 1; k >= 0 ; k--) {
            int idx = nonfixed[k];
            if (x[idx].isFixed()) {
                int fixedValue = x[idx].min();
                for (int j = 0; j < k; j++) {
                    x[nonfixed[j]].remove(fixedValue);
                }
                // Swap
                s--;
                nonfixed[k] = nonfixed[s];
                nonfixed[s] = idx;
            }
        }
        nbNonFixed.setValue(s);
    }

    /**
     * Sends the JSON representation of the domains to the AI model and retrieves the response.
     * @param json JSONObject representing the domains of the variables
     * @return JSONObject containing the response from the AI model
     * @throws IOException
     */
    private static synchronized JSONObject sendJsonAndGetResponse(JSONObject json) throws IOException {
        String jsonStr = json.toString();
        SocketManager mgr = SocketManager.getInstance();

        // Send and receive
        String respLine = mgr.sendAndReceive(jsonStr);
        if (respLine == null) {
            throw new IOException("No response from AI model");
        }
        return new JSONObject(respLine);
    }

}
