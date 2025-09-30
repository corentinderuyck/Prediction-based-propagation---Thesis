    package org.maxicp.RunXCSP3;

    import org.maxicp.cp.engine.constraints.AtLeastNValueDC;
    import org.maxicp.modeling.algebra.bool.Eq;
    import org.maxicp.modeling.algebra.bool.NotEq;
    import org.maxicp.modeling.algebra.integer.IntExpression;
    import org.maxicp.modeling.xcsp3.XCSP3;
    import org.maxicp.search.DFSearch;
    import org.maxicp.search.SearchStatistics;
    import org.maxicp.util.exception.InconsistencyException;
    import org.newsclub.net.unix.AFUNIXSocket;
    import org.newsclub.net.unix.AFUNIXSocketAddress;

    import java.io.*;
    import java.nio.charset.StandardCharsets;
    import java.util.*;
    import java.util.concurrent.*;
    import java.util.function.Supplier;

    import static org.maxicp.modeling.xcsp3.XCSP3.load;
    import static org.maxicp.search.Searches.*;

    public class RunXSCP3 {

        private static AFUNIXSocket socket;

        private static class RunResult {
            SearchStatistics stats;
            long executionTimeMillis;

            RunResult(SearchStatistics stats, long executionTimeMillis) {
                this.stats = stats;
                this.executionTimeMillis = executionTimeMillis;
            }
        }


        // -------------- Code to run XCSP3 instances with AI model support ----------------
        // Need to set the var USE_AI_MODEL to true in the AllDifferentDC.java file

        // ---- Use the server with the AI model -----

        public static void startServer() throws InterruptedException, IOException {
            // Start the Python server
            Process pythonServer = new ProcessBuilder()
                    .command("bash", "-c", "source ../python/env/bin/activate && python3 ../python/use_model_in_java_tiny.py")
                    .inheritIO()
                    .start();

            // Wait to ensure the server is fully initialized
            Thread.sleep(20000);

            File socketFile = new File("/tmp/unix_socket_predictor");
            socket = AFUNIXSocket.newInstance();
            socket.connect(new AFUNIXSocketAddress(socketFile));
        }

        public static void stopServer() throws IOException, InterruptedException {
            if (socket != null && !socket.isClosed()) {
                OutputStream out = socket.getOutputStream();
                out.write("{\"kill\": true}\n".getBytes(StandardCharsets.UTF_8));
                out.flush();
                socket.close();
            }

            Thread.sleep(3000);
        }

        public static void changeThreshold(float threshold) throws IOException, InterruptedException {
            if (socket == null || socket.isClosed()) {
                throw new IllegalStateException("Socket not connected");
            }

            OutputStream out = socket.getOutputStream();
            String json = String.format(Locale.US, "{\"threshold\": %.2f}\n", threshold);
            out.write(json.getBytes(StandardCharsets.UTF_8));
            out.flush();

            // Wait for the server to process the change
            Thread.sleep(10000);
        }

        // ---- Run instances with AI model support -----

        /**
         * Runs a single XCSP3 instance with AI model support.
         * @param instanceFile the path to the XML instance file
         * @return
         * @throws Exception
         */
        public static RunResult runOneInstanceAI(String instanceFile) throws Exception {

            long start = System.currentTimeMillis();

            XCSP3.XCSP3LoadedInstance instance = load(instanceFile);
            IntExpression[] q = instance.decisionVars();

            // first fail branching (minimal domain size)
            Supplier<Runnable[]> branching = () -> {
                if (Thread.currentThread().isInterrupted()) {
                    System.out.println("Interruption detected during branching. Stopping search.");
                    throw new RuntimeException("Interrupted");
                }

                IntExpression qs = selectMin(q,
                        qi -> qi.size() > 1,
                        qi -> qi.size());

                if (qs == null) return EMPTY;
                int v = qs.min();

                Runnable left = () -> instance.md().add(new Eq(qs, v));
                Runnable right = () -> instance.md().add(new NotEq(qs, v));
                return branch(left, right);
            };

            final SearchStatistics[] result = new SearchStatistics[1];

            try {
                // Run the XCSP3 instance with the AI model
                instance.md().runCP(cp -> {
                    DFSearch search = cp.dfSearch(branching);
                    SearchStatistics stats = search.solve();
                    System.out.println(stats);
                    result[0] = stats;
                });
            } catch (InconsistencyException e) {
                System.out.println("Inconsistency detected");
                e.printStackTrace();
                result[0] = new SearchStatistics();
            }

            long end = System.currentTimeMillis();

            return new RunResult(result[0], end - start);
        }

        /**
         * Runs a single XCSP3 instance with a different threshold for the AI model.
         * Save the statistics to a CSV file.
         * @param instanceFile the path to the XML instance file
         * @throws Exception
         */
        private static void runOneInstanceDifferentThresholdAI(String instanceFile) throws Exception {

            String fileName = "../data/stats.csv";
            float[] thresholds = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
            for (float threshold : thresholds) {
                // change threshold in the Python server
                changeThreshold(threshold);

                System.out.println("Running instance: " + instanceFile + " with threshold: " + threshold);
                RunResult runResult = runOneInstanceAI(instanceFile);
                System.out.println("Finished instance: " + instanceFile + " with threshold: " + threshold);
                saveStatistics(fileName, instanceFile, runResult, threshold);
            }

        }

        /**
         * Runs all XCSP3 instances in a specified folder with different thresholds for the AI model.
         * Save the statistics to a CSV file.
         * @param folder the folder containing the XML instances
         * @throws Exception
         */
        private static void runAllInstanceFolderAI(String folder) throws Exception {
            // start the Python server
            startServer();


            File dir = new File(folder);
            if (!dir.exists() || !dir.isDirectory()) {
                throw new IllegalArgumentException("Invalid folder: " + folder);
            }

            File[] files = dir.listFiles((d, name) -> name.toLowerCase().endsWith(".xml"));
            if (files == null || files.length == 0) {
                System.out.println("No XML instances found in folder: " + folder);
                return;
            }

            for (File file : files) {
                try {
                    Thread.sleep(5000);
                    runOneInstanceDifferentThresholdAI(file.getAbsolutePath());
                    Thread.sleep(5000);
                } catch (Exception e) {
                    System.err.println("Error running instance: " + file.getName());
                    e.printStackTrace();
                }
            }

            // stop the Python server
            stopServer();
        }


        // -------------- Code use to generate the data to train the AI model ----------------
        // Need to set the var USE_AI_MODEL to false in the AllDifferentDC.java file
        // Need to set the var saveState to true in the AtLeastNValueDC.java file
        // Need to specify if one file is used for all instances or one file per instance in the AtLeastNValueDC.java file

        /**
         * Runs a single XCSP3 instance and returns the search statistics.
         * @param instanceFile the path to the XML instance file
         * @param onefile: if true, the propagation is save in one file, otherwise in a separate file for each instance
         * @throws Exception
         */
        private static RunResult runOneInstance(String instanceFile, boolean onefile) throws Exception {

            // Set the current instance name for AtLeastNValueDC
            if (onefile) {
                AtLeastNValueDC.setCurrentInstanceName("train_data");
            } else {
                String fileName = instanceFile.substring(instanceFile.lastIndexOf("/") + 1);
                if (!fileName.endsWith(".xml")) {
                    throw new IllegalArgumentException("Instance file must be an XML file: " + instanceFile);
                }
                String instanceName = fileName.substring(0, fileName.length() - 4);
                AtLeastNValueDC.setCurrentInstanceName(instanceName);
            }

            long start = System.currentTimeMillis();

            XCSP3.XCSP3LoadedInstance instance = load(instanceFile);
            IntExpression[] q = instance.decisionVars();

            // first fail branching (minimal domain size)
            Supplier<Runnable[]> branching = () -> {
                if (Thread.currentThread().isInterrupted()) {
                    System.out.println("Interruption detected during branching. Stopping search.");
                    throw new RuntimeException("Interrupted");
                }

                IntExpression qs = selectMin(q,
                        qi -> qi.size() > 1,
                        qi -> qi.size());

                if (qs == null) return EMPTY;
                int v = qs.min();

                Runnable left = () -> instance.md().add(new Eq(qs, v));
                Runnable right = () -> instance.md().add(new NotEq(qs, v));
                return branch(left, right);
            };

            final SearchStatistics[] result = new SearchStatistics[1];

            instance.md().runCP(cp -> {
                DFSearch search = cp.dfSearch(branching);
                SearchStatistics stats = search.solve();
                System.out.println(stats);
                result[0] = stats;
            });

            long end = System.currentTimeMillis();
            return new RunResult(result[0], end - start);

        }

        /**
         * Runs all XCSP3 instances in a specified folder.
         * Use to record the state before and after the propagation of the All Different constraint.
         * @param folder: the folder containing the XML instances
         * @param oneFile: if true, the propagation is save in one file, otherwise in a separate file for each instance
         * @throws Exception
         */
        public static void runAllInstances(String folder, boolean oneFile) throws Exception {
            File dir = new File(folder);
            if (!dir.exists() || !dir.isDirectory()) {
                throw new IllegalArgumentException("Invalid folder: " + folder);
            }

            File[] files = dir.listFiles((d, name) -> name.toLowerCase().endsWith(".xml"));
            if (files == null || files.length == 0) {
                System.out.println("No XML instances found in folder: " + folder);
                return;
            }

            ExecutorService executor = Executors.newSingleThreadExecutor();

            for (File file : files) {

                Future<?> future = executor.submit(() -> {
                    try {
                        System.out.println("Running instance: " + file.getName());
                        RunResult result = runOneInstance(file.getAbsolutePath(), oneFile);
                        System.out.println("Finished instance: " + file.getName());
                        saveStatistics("run_without_ai_train.csv", file.getName(), result, 0.0f);
                    } catch (Exception e) {
                        // Do nothing
                    }
                });

                try {
                    future.get();
                } catch (Exception e) {
                    // Do nothing
                }

            }

            executor.shutdown();
        }

        // ---- Use to get statistics of the search and save them to a CSV file -----
        // With AI model or without AI model support (for comparison)

        /**
         * Saves the search statistics to a CSV file.
         * The file is created if it does not exist, and the header is written only once.
         * @param csvFile the path to the CSV file
         * @param instanceName the name of the instance
         * @param runResult the result of the run
         * @param threshold (optional) threshold used for the AI model
         */
        private static void saveStatistics(String csvFile, String instanceName, RunResult runResult, float threshold) {
            boolean fileExists = new File(csvFile).exists();

            try (FileWriter fw = new FileWriter(csvFile, true);
                 BufferedWriter bw = new BufferedWriter(fw);
                 PrintWriter out = new PrintWriter(bw)) {

                if (!fileExists) {
                    out.println("Instance,Nodes,Failures,Solutions,Completed,ExecutionTime(ms),Threshold");
                }

                String baseName = instanceName.endsWith(".xml") ?
                        instanceName.substring(0, instanceName.length() - 4) : instanceName;

                out.printf(Locale.US, "%s,%d,%d,%d,%b,%d,%.2f%n",
                        baseName,
                        runResult.stats.numberOfNodes(),
                        runResult.stats.numberOfFailures(),
                        runResult.stats.numberOfSolutions(),
                        runResult.stats.isCompleted(),
                        runResult.executionTimeMillis,
                        threshold);


                System.out.println("Statistics saved to: " + csvFile);

            } catch (IOException e) {
                System.err.println("Failed to save statistics for " + instanceName);
                e.printStackTrace();
            }
        }

        public static void main(String[] args) throws Exception {
            runAllInstanceFolderAI("filtered_xml_instances_test");
        }
    }