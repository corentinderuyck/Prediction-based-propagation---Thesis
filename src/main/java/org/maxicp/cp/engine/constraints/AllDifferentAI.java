package org.maxicp.cp.engine.constraints;

import org.json.JSONArray;
import org.json.JSONObject;
import org.maxicp.cp.engine.core.AbstractCPConstraint;
import org.maxicp.cp.engine.core.CPIntVar;
import org.maxicp.state.StateInt;
import org.maxicp.util.exception.InconsistencyException;
import org.newsclub.net.unix.AFUNIXSocket;
import org.newsclub.net.unix.AFUNIXSocketAddress;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.stream.IntStream;

public class AllDifferentAI extends AbstractCPConstraint {
    private CPIntVar[] x;

    private static final String SOCKET_PATH = "/tmp/unix_socket_predictor";
    private static AFUNIXSocket socket;
    private static BufferedReader reader;
    private static OutputStream out;


    public AllDifferentAI(CPIntVar... x) {
        super(x[0].getSolver());
        this.x = x;
    }

    @Override
    public void post() {
        for (int k = 0; k < x.length; k++) {
            if (!x[k].isFixed()) {
                x[k].propagateOnDomainChange(this);
            }
        }

        //checker();

    }

    @Override
    public void propagate() {

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
            JSONObject response = sendJsonAndGetResponse(graph);

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
            try {
                closeSocket();
            } catch (IOException ex) {
                // ignore
            }
        }

    }

    public void checker() {
        for (int i = 0; i < x.length; i++) {
            if (x[i].isFixed()) {
                int fixedValue = x[i].min();

                for (int j = 0; j < x.length; j++) {
                    if (i != j) {
                        x[j].remove(fixedValue);
                    }
                }
            }
        }
    }

    private static void openSocketIfNeeded() throws IOException {
        if (socket == null || socket.isClosed() || !socket.isConnected()) {
            File socketFile = new File(SOCKET_PATH);
            socket = AFUNIXSocket.newInstance();
            socket.connect(new AFUNIXSocketAddress(socketFile));
            reader = new BufferedReader(new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8));
            out = socket.getOutputStream();
        }
    }

    private static void closeSocket() throws IOException {
        if (reader != null) reader.close();
        if (out != null) out.close();
        if (socket != null) socket.close();
        reader = null;
        out = null;
        socket = null;
    }

    /**
     * Sends the JSON representation of the domains to the AI model and retrieves the response.
     * @param json
     * @return JSONObject containing the response from the AI model
     * @throws IOException
     */
    private static synchronized JSONObject sendJsonAndGetResponse(JSONObject json) throws IOException {
        openSocketIfNeeded();

        // Write JSON request
        String jsonStr = json.toString() + "\n";
        out.write(jsonStr.getBytes(StandardCharsets.UTF_8));
        out.flush();

        // Read JSON response
        String line = reader.readLine();
        if (line == null) {
            throw new IOException("No response from AI model");
        }
        return new JSONObject(line);
    }

}
