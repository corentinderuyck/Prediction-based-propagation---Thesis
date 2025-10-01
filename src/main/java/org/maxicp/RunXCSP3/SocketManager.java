package org.maxicp.RunXCSP3;

import org.newsclub.net.unix.AFUNIXSocket;
import org.newsclub.net.unix.AFUNIXSocketAddress;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;

public class SocketManager {
    private static final String SOCKET_PATH = "/tmp/unix_socket_predictor";
    private static SocketManager instance;

    private AFUNIXSocket socket;
    private BufferedReader reader;
    private PrintWriter writer;
    private final Object lock = new Object();

    // config
    private int connectSleepMs = 2000;      // delay between reconnect attempts
    private int readTimeoutMs = 30000;      // socket SO_TIMEOUT for reads

    private SocketManager() { }

    public static synchronized SocketManager getInstance() {
        if (instance == null) instance = new SocketManager();
        return instance;
    }

    private void ensureConnected() throws IOException {
        if (socket != null && !socket.isClosed() && socket.isConnected() && reader != null && writer != null) {
            return;
        }

        // (re)connect
        File socketFile = new File(SOCKET_PATH);
        socket = AFUNIXSocket.newInstance();
        socket.connect(new AFUNIXSocketAddress(socketFile));
        socket.setSoTimeout(readTimeoutMs);
        reader = new BufferedReader(new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8));
        writer = new PrintWriter(new OutputStreamWriter(socket.getOutputStream(), StandardCharsets.UTF_8), true);
    }

    /**
     * Sends a JSON line to the socket and waits for a single line response.
     * @param jsonLine JSON string (without newline)
     * @return response line from the socket
     * @throws IOException
     */
    public String sendAndReceive(String jsonLine) throws IOException {
        synchronized (lock) {
            try {
                ensureConnected();
                // write
                writer.print(jsonLine.endsWith("\n") ? jsonLine : (jsonLine + "\n"));
                writer.flush();

                // read one line response (blocks up to readTimeoutMs)
                String response = reader.readLine();
                if (response == null) {
                    // remote closed stream
                    throw new IOException("Remote socket closed (null response)");
                }
                return response;
            } catch (IOException e) {
                // attempt a single reconnect and retry once
                closeQuietly();
                try {
                    TimeUnit.MILLISECONDS.sleep(connectSleepMs);
                    ensureConnected();
                    writer.print(jsonLine.endsWith("\n") ? jsonLine : (jsonLine + "\n"));
                    writer.flush();
                    String response = reader.readLine();
                    if (response == null) throw new IOException("Remote socket closed on retry");
                    return response;
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    throw new IOException("Interrupted during reconnect", ie);
                }
            }
        }
    }

    public void sendNoResponse(String jsonLine) throws IOException {
        synchronized (lock) {
            ensureConnected();
            writer.print(jsonLine.endsWith("\n") ? jsonLine : (jsonLine + "\n"));
            writer.flush();
        }
    }

    public synchronized void closeQuietly() {
        try { if (reader != null) reader.close(); } catch (IOException ignored) {}
        try { if (writer != null) writer.close(); } catch (Exception ignored) {}
        try { if (socket != null) socket.close(); } catch (IOException ignored) {}
        reader = null;
        writer = null;
        socket = null;
    }

    public void setReadTimeoutMs(int ms) throws IOException {
        this.readTimeoutMs = ms;
        if (socket != null && !socket.isClosed()) socket.setSoTimeout(ms);
    }
}
