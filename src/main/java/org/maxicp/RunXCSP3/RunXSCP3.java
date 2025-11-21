    package org.maxicp.RunXCSP3;

    import org.maxicp.cp.engine.constraints.AllDifferentAI;
    import org.maxicp.cp.engine.constraints.AtLeastNValueDC;
    import org.maxicp.modeling.algebra.bool.Eq;
    import org.maxicp.modeling.algebra.bool.NotEq;
    import org.maxicp.modeling.algebra.integer.IntExpression;
    import org.maxicp.modeling.symbolic.Objective;
    import org.maxicp.modeling.xcsp3.XCSP3;
    import org.maxicp.search.DFSearch;
    import org.maxicp.search.SearchStatistics;
    import org.maxicp.util.exception.InconsistencyException;
    import org.maxicp.util.exception.NotImplementedException;
    import org.maxicp.util.exception.NotYetImplementedException;

    import javax.json.JsonObject;
    import java.io.*;
    import java.util.*;
    import java.util.concurrent.*;
    import java.util.concurrent.atomic.AtomicInteger;
    import java.util.function.Supplier;

    import static org.maxicp.modeling.xcsp3.XCSP3.load;
    import static org.maxicp.search.Searches.*;

    public class RunXSCP3 {

        private static Process pythonServerProcess = null;

        private static class RunResult {
            SearchStatistics stats;
            long executionTimeMillis;
            long javaTimeMillis;
            long pythonTimeMillis;
            int nbCallPropagate;

            RunResult(SearchStatistics stats, long executionTimeMillis, long javaTimeMillis, long pythonTimeMillis, int nbCallPropagate) {
                this.stats = stats;
                this.executionTimeMillis = executionTimeMillis;
                this.javaTimeMillis = javaTimeMillis;
                this.pythonTimeMillis = pythonTimeMillis;
                this.nbCallPropagate = nbCallPropagate;
            }

            @Override
            public String toString() {
                String output = String.format("Nodes: %d\nFailures: %d\nSolutions: %d\nCompleted: %b\nTotal Execution Time (ms): %d\nJava Execution Time (ms): %d\nPython AI Execution Time (ms): %d\nNumber of Calls to Propagate: %d",
                         stats.numberOfNodes(),
                         stats.numberOfFailures(),
                         stats.numberOfSolutions(),
                         stats.isCompleted(),
                         executionTimeMillis,
                         javaTimeMillis,
                         pythonTimeMillis,
                         nbCallPropagate);
                return output;
            }
        }

        private static class RunResultOptimization {
            SearchStatistics stats;
            int bestValue;
            long executionTimeMillis;
            long javaTimeMillis;
            long pythonTimeMillis;
            int nbCallPropagate;
            boolean isMinimization;

            RunResultOptimization(SearchStatistics stats, long executionTimeMillis, long javaTimeMillis, long pythonTimeMillis, int nbCallPropagate, int bestValue, boolean isMinimization) {
                this.stats = stats;
                this.executionTimeMillis = executionTimeMillis;
                this.javaTimeMillis = javaTimeMillis;
                this.pythonTimeMillis = pythonTimeMillis;
                this.nbCallPropagate = nbCallPropagate;
                this.bestValue = bestValue;
                this.isMinimization = isMinimization;
            }

            @Override
            public String toString() {
                String output = String.format("Nodes: %d\nFailures: %d\nSolutions: %d\nCompleted: %b\nTotal Execution Time (ms): %d\nJava Execution Time (ms): %d\nPython AI Execution Time (ms): %d\nNumber of Calls to Propagate: %d\nBest Objective Value: %d\nisMinimization: %b",
                        stats.numberOfNodes(),
                        stats.numberOfFailures(),
                        stats.numberOfSolutions(),
                        stats.isCompleted(),
                        executionTimeMillis,
                        javaTimeMillis,
                        pythonTimeMillis,
                        nbCallPropagate,
                        bestValue,
                        isMinimization);
                return output;
            }
        }


        // -------------- Code to run XCSP3 instances with AI model support ----------------
        // Need to set the var USE_AI_MODEL to true in the AllDifferentDC.java file

        // ---- Use the server with the AI model -----

        public static void startServer() throws InterruptedException, IOException {
            pythonServerProcess = new ProcessBuilder()
                    .command("bash", "-c", "source ../python/env/bin/activate && python3 ../python/use_model_in_java.py")
                    .inheritIO()
                    .start();

            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                System.out.println("Shutting down Python server...");
                if (pythonServerProcess != null && pythonServerProcess.isAlive()) {
                    try {
                        stopServer();
                    } catch (Exception e) {
                        System.err.println("Error during shutdown: " + e.getMessage());
                    }
                    pythonServerProcess.destroyForcibly();
                }
            }));

            Thread.sleep(20000);

            try {
                SocketManager.getInstance().sendNoResponse("{\"ping\": true}");
            } catch (IOException e) {
                System.err.println("Unable to connect to python socket after starting server: " + e.getMessage());
                throw e;
            }
        }

        public static void stopServer() throws IOException, InterruptedException {
            try {
                SocketManager.getInstance().sendNoResponse("{\"kill\": true}");
                Thread.sleep(1000);
            } catch (IOException e) {
                System.err.println("Warning: could not send kill to python server: " + e.getMessage());
            } finally {
                SocketManager.getInstance().closeQuietly();
            }

            if (pythonServerProcess != null && pythonServerProcess.isAlive()) {
                pythonServerProcess.destroy();
                if (!pythonServerProcess.waitFor(5, TimeUnit.SECONDS)) {
                    System.err.println("Python process did not terminate, forcing...");
                    pythonServerProcess.destroyForcibly();
                }
            }
        }

        public static void changeThreshold(float threshold) throws IOException, InterruptedException {
            String json = String.format(Locale.US, "{\"threshold\": %.2f}", threshold);
            try {
                SocketManager.getInstance().sendNoResponse(json);
            } catch (IOException e) {
                throw e;
            }

            Thread.sleep(2000);
        }

        public static long getTimePython() throws IOException {
            String json = "{\"time\": true}";
            String line;
            try {
                line = SocketManager.getInstance().sendAndReceive(json);
            } catch (IOException e) {
                throw e;
            }

            if (line != null) {
                JsonObject response = javax.json.Json.createReader(new StringReader(line)).readObject();
                return response.getJsonNumber("totalTime").longValue();
            } else {
                System.out.println("No response from AI model");
                return -1;
            }
        }

        public static void resetTimeAI() throws IOException, InterruptedException {
            String json = "{\"reset_time\": true}";
            try {
                SocketManager.getInstance().sendNoResponse(json);
            } catch (IOException e) {
                throw e;
            }

            Thread.sleep(2000);
        }


        // ---- Run instances with AI model support -----

        /**
         * Runs a single XCSP3 instance with AI model support.
         * @param instanceFile the path to the XML instance file
         * @return
         * @throws Exception
         */
        public static RunResult runOneInstanceAI(String instanceFile) throws Exception {

            XCSP3.XCSP3LoadedInstance instance = load(instanceFile);
            IntExpression[] q = instance.decisionVars();

            // first fail branching (minimal domain size)
            Supplier<Runnable[]> branching = () -> {

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

            AllDifferentAI.nbCallPropagate = 0;     // Count the number of calls to the propagate method

            resetTimeAI();      // Reset the time spent in Python

            Stopwatch allExecutionTimer = new Stopwatch();      // Count the total execution time including Java and Python (and socket communication)
            allExecutionTimer.reset();
            allExecutionTimer.start();

            GlobalTimers.allJavaTimer.reset();        // Count the total execution time in Java only (without Python and socket communication)
            GlobalTimers.allJavaTimer.start();

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

            allExecutionTimer.pause();
            long totalExecutionTime = allExecutionTimer.getElapsedTimeMillis();

            GlobalTimers.allJavaTimer.pause();
            long totalJavaTime = GlobalTimers.allJavaTimer.getElapsedTimeMillis();

            long totalPythonTime = getTimePython();

            int nbCallPropagate = AllDifferentAI.nbCallPropagate;

            return new RunResult(result[0], totalExecutionTime, totalJavaTime, totalPythonTime, nbCallPropagate);
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

            final Set<String> INSTANCES_WITHOUT_SOLUTIONS = Set.of(
                    "SchurrLemma-015-9-mod",
                    "SchurrLemma-020-9-mod",
                    "SchurrLemma-012-9-mod",
                    "SchurrLemma-050-9-mod",
                    "SchurrLemma-030-9-mod",
                    "SchurrLemma-100-9-mod",
                    "Langford-4-09",
                    "ColouredQueens-03",
                    "Subisomorphism-g16-g46",
                    "Langford-2-06",
                    "OpenStacks-m1-wbop-30-10-1"
            );

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

                String fileName = file.getName();
                String instanceName = fileName.substring(0, fileName.lastIndexOf('.'));

                if (INSTANCES_WITHOUT_SOLUTIONS.contains(instanceName)) {
                    System.out.println("Skipping instance without solutions: " + file.getName());
                    continue;
                }

                try {
                    runOneInstanceDifferentThresholdAI(file.getAbsolutePath());
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

            AtLeastNValueDC.nbCallPropagate = 0;     // Count the number of calls to the propagate method

            Stopwatch allExecutionTimer = new Stopwatch();
            allExecutionTimer.reset();
            allExecutionTimer.start();

            XCSP3.XCSP3LoadedInstance instance = load(instanceFile);
            IntExpression[] q = instance.decisionVars();

            // first fail branching (minimal domain size)
            Supplier<Runnable[]> branching = () -> {

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

            allExecutionTimer.pause();
            long totalExecutionTime = allExecutionTimer.getElapsedTimeMillis();

            int nbCallPropagate = AtLeastNValueDC.nbCallPropagate;

            return new RunResult(result[0], totalExecutionTime, totalExecutionTime, 0L, nbCallPropagate);

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
                String baseName = file.getName().endsWith(".xml") ?
                        file.getName().substring(0, file.getName().length() - 4) : file.getName();

                if (baseName.equals("RubiksCube")) {
                    System.out.println("Skipping instance: " + file.getName());
                    continue;
                }

                Future<?> future = executor.submit(() -> {
                    try {
                        System.out.println("Running instance: " + file.getName());
                        RunResult result = runOneInstance(file.getAbsolutePath(), oneFile);
                        System.out.println("Finished instance: " + file.getName());
                        saveStatistics("run_without_ai.csv", file.getName(), result, 0.0f);
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
                    out.println("Instance,Nodes,Failures,Solutions,Completed,TotalExecutionTimeMillis,JavaExecutionTimeMillis,PythonAIExecutionTimeMillis,NumberOfCallsToPropagate,Threshold");
                }

                String baseName = instanceName.endsWith(".xml") ?
                        instanceName.substring(0, instanceName.length() - 4) : instanceName;

                out.printf(Locale.US, "%s,%d,%d,%d,%b,%d,%d,%d,%d,%.2f%n",
                        baseName,
                        runResult.stats.numberOfNodes(),
                        runResult.stats.numberOfFailures(),
                        runResult.stats.numberOfSolutions(),
                        runResult.stats.isCompleted(),
                        runResult.executionTimeMillis,
                        runResult.javaTimeMillis,
                        runResult.pythonTimeMillis,
                        runResult.nbCallPropagate,
                        threshold);


                System.out.println("Statistics saved to: " + csvFile);

            } catch (IOException e) {
                System.err.println("Failed to save statistics for " + instanceName);
                e.printStackTrace();
            }
        }

        /**
         * Saves the search statistics to a CSV file.
         * The file is created if it does not exist, and the header is written only once.
         * @param csvFile the path to the CSV file
         * @param instanceName the name of the instance
         * @param runResult the result of the run
         * @param threshold (optional) threshold used for the AI model
         */
        private static void saveStatisticsOptimization(String csvFile, String instanceName, RunResultOptimization runResult, float threshold) {
            boolean fileExists = new File(csvFile).exists();

            try (FileWriter fw = new FileWriter(csvFile, true);
                 BufferedWriter bw = new BufferedWriter(fw);
                 PrintWriter out = new PrintWriter(bw)) {

                if (!fileExists) {
                    out.println("Instance,Nodes,Failures,Solutions,Completed,TotalExecutionTimeMillis,JavaExecutionTimeMillis,PythonAIExecutionTimeMillis,NumberOfCallsToPropagate,Threshold,BestObjectiveValue,isMinimization");
                }

                String baseName = instanceName.endsWith(".xml") ?
                        instanceName.substring(0, instanceName.length() - 4) : instanceName;

                out.printf(Locale.US, "%s,%d,%d,%d,%b,%d,%d,%d,%d,%.2f,%d,%b%n",
                        baseName,
                        runResult.stats.numberOfNodes(),
                        runResult.stats.numberOfFailures(),
                        runResult.stats.numberOfSolutions(),
                        runResult.stats.isCompleted(),
                        runResult.executionTimeMillis,
                        runResult.javaTimeMillis,
                        runResult.pythonTimeMillis,
                        runResult.nbCallPropagate,
                        threshold,
                        runResult.bestValue,
                        runResult.isMinimization);


                System.out.println("Statistics saved to: " + csvFile);

            } catch (IOException e) {
                System.err.println("Failed to save statistics for " + instanceName);
                e.printStackTrace();
            }
        }



        // -------------- Code to run optimization instances ----------------

        private static void runOptimizationInstancesFolder(String folder) {
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
                    System.out.println("\nRunning optimization instance: " + file.getName());
                    RunResultOptimization results = runOptimizationInstance(file.getAbsolutePath());
                    System.out.println(results);
                    saveStatisticsOptimization("instances_optimization_inverse.csv", file.getName(), results, 0.0f);
                } catch (Exception e) {
                    System.err.println("Error running instance: " + file.getName());
                    e.printStackTrace();
                }
            }
        }

        private static RunResultOptimization runOptimizationInstance(String instanceFile) throws Exception {

            AtLeastNValueDC.nbCallPropagate = 0;     // Count the number of calls to the propagate method

            Stopwatch allExecutionTimer = new Stopwatch();
            allExecutionTimer.reset();
            allExecutionTimer.start();

            XCSP3.XCSP3LoadedInstance instance = load(instanceFile);
            IntExpression[] q = instance.decisionVars();

            Objective obj = instance.objective();
            if (obj == null) {
                throw new IllegalStateException("Instance does not define an objective");
            }
            boolean isMinimization = instance.isMinimization();

            AtomicInteger bestValue = new AtomicInteger(isMinimization ? Integer.MAX_VALUE : Integer.MIN_VALUE);

            // first fail branching (minimal domain size)
            Supplier<Runnable[]> branching = () -> {

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
                search.onSolution(() -> {
                    String s = obj.toString();
                    // Extract the integer value from the objective string
                    int value = Integer.parseInt(s.replaceAll("\\D+", ""));
                    bestValue.set(value);
                });
                SearchStatistics stats = search.optimize(obj);
                result[0] = stats;
            });

            allExecutionTimer.pause();
            long totalExecutionTime = allExecutionTimer.getElapsedTimeMillis();

            int nbCallPropagate = AtLeastNValueDC.nbCallPropagate;

            return new RunResultOptimization(result[0], totalExecutionTime, totalExecutionTime, 0L, nbCallPropagate, bestValue.get(), isMinimization);
        }

        private static RunResultOptimization runOptimizationInstanceAI(String instanceFile) throws Exception {

            AllDifferentAI.nbCallPropagate = 0;     // Count the number of calls to the propagate method

            Stopwatch allExecutionTimer = new Stopwatch();
            allExecutionTimer.reset();
            allExecutionTimer.start();

            XCSP3.XCSP3LoadedInstance instance = load(instanceFile);
            IntExpression[] q = instance.decisionVars();

            Objective obj = instance.objective();
            if (obj == null) {
                throw new IllegalStateException("Instance does not define an objective");
            }
            boolean isMinimization = instance.isMinimization();

            AtomicInteger bestValue = new AtomicInteger(isMinimization ? Integer.MAX_VALUE : Integer.MIN_VALUE);

            // first fail branching (minimal domain size)
            Supplier<Runnable[]> branching = () -> {

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
                search.onSolution(() -> {
                    String s = obj.toString();
                    // Extract the integer value from the objective string
                    int value = Integer.parseInt(s.replaceAll("\\D+", ""));
                    bestValue.set(value);
                });
                SearchStatistics stats = search.optimize(obj);
                result[0] = stats;
            });

            allExecutionTimer.pause();
            long totalExecutionTime = allExecutionTimer.getElapsedTimeMillis();

            int nbCallPropagate = AllDifferentAI.nbCallPropagate;

            return new RunResultOptimization(result[0], totalExecutionTime, totalExecutionTime, 0L, nbCallPropagate, bestValue.get(), isMinimization);
        }

        private static void runOptimizationInstancesFolderAI(String folder) throws IOException, InterruptedException {
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
                    System.out.println("Running optimization instance: " + file.getName());
                    runOneInstanceOptimizationDifferentThresholdAI(file.getAbsolutePath());
                } catch (Exception e) {
                    System.err.println("Error running instance: " + file.getName());
                    e.printStackTrace();
                }
            }

            // stop the Python server
            stopServer();
        }

        private static void runOneInstanceOptimizationDifferentThresholdAI(String instanceFile) throws Exception {

            String fileName = "../data/stats_opti.csv";
            float[] thresholds = {0.01f,0.05f,0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
            for (float threshold : thresholds) {
                // change threshold in the Python server
                changeThreshold(threshold);

                System.out.println("Running instance: " + instanceFile + " with threshold: " + threshold);
                RunResultOptimization runResult = runOptimizationInstanceAI(instanceFile);
                System.out.println("Finished instance: " + instanceFile + " with threshold: " + threshold);
                saveStatisticsOptimization(fileName, instanceFile, runResult, threshold);
            }

        }



        public static void main(String[] args) throws Exception {

            try {
                runAllInstanceFolderAI("filtered_xml_instances_test");
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                try {
                    if (pythonServerProcess != null && pythonServerProcess.isAlive()) {
                        stopServer();
                    }
                } catch (Exception e) {
                    System.err.println("Failed to stop server: " + e.getMessage());
                }
            }

            /*
            // run optimization instances
            try {
                runOptimizationInstancesFolderAI("COP_instances");
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                try {
                    if (pythonServerProcess != null && pythonServerProcess.isAlive()) {
                        stopServer();
                    }
                } catch (Exception e) {
                    System.err.println("Failed to stop server: " + e.getMessage());
                }
            }

            */

            //runAllInstances("filtered_xml_instances_test", false);
            //runOptimizationInstancesFolder("COP_instances_inverse");

        }
    }