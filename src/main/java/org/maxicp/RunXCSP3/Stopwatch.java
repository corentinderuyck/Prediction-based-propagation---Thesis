package org.maxicp.RunXCSP3;

public class Stopwatch {
    private long startTime = 0;
    private long accumulated = 0;
    private boolean running = false;

    public void start() {
        if (!running) {
            startTime = System.nanoTime();
            running = true;
        }
    }

    public void pause() {
        if (running) {
            accumulated += System.nanoTime() - startTime;
            running = false;
        }
    }

    public void reset() {
        startTime = 0;
        accumulated = 0;
        running = false;
    }

    public long getElapsedTimeMillis() {
        return (running ? accumulated + (System.nanoTime() - startTime) : accumulated) / 1_000_000;
    }
}
