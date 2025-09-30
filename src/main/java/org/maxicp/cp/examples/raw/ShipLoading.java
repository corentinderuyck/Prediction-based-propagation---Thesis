/*
 * MaxiCP is under MIT License
 * Copyright (c)  2024 UCLouvain
 *
 */

package org.maxicp.cp.examples.raw;

import org.maxicp.cp.CPFactory;
import org.maxicp.cp.engine.constraints.scheduling.CPCumulFunction;
import org.maxicp.cp.engine.core.CPIntVar;
import org.maxicp.cp.engine.core.CPIntervalVar;
import org.maxicp.cp.engine.core.CPSolver;
import org.maxicp.search.DFSearch;
import org.maxicp.search.Objective;
import org.maxicp.search.SearchStatistics;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

import static org.maxicp.cp.CPFactory.*;
import static org.maxicp.search.Searches.and;
import static org.maxicp.search.Searches.firstFail;

/**
 * Ship Loading Problem.
 *
 * The problem is to find a schedule that minimizes the time to unload and
 * to load a ship. The work contains a set of 34 elementary tasks. Each task has to be handled by
 * a given number of people and during a given period of time. For each task, only
 * the associated surface is known (i.e., the product of the task duration by the needed amount of
 * resource).
 *
 * This problem was described in the paper:
 * Aggoun, A., & Beldiceanu, N. (1993). Extending CHIP in order to solve complex scheduling and placement problems.
 * Mathematical and computer modelling, 17(7), 57-73.
 *
 * @author Roger Kameugne, Pierre Schaus
 */
public class ShipLoading {


    public ShipLoading(ShipLoadingInstance data) throws Exception {


        CPSolver cp = CPFactory.makeSolver();

        CPIntervalVar[] intervals = new CPIntervalVar[data.nbTasks];
        CPIntVar[] starts = new CPIntVar[data.nbTasks];
        CPIntVar[] ends = new CPIntVar[data.nbTasks];
        CPIntVar[] length = new CPIntVar[data.nbTasks];
        CPIntVar[] height = new CPIntVar[data.nbTasks];

        CPCumulFunction resource = flat();
        for (int i = 0; i < data.nbTasks; i++) {
            CPIntervalVar interval = makeIntervalVar(cp);
            interval.setEndMax(data.horizon);
            interval.setLengthMin(1); // remove
            interval.setLengthMax(data.sizes[i]);
            interval.setPresent();
            intervals[i] = interval;

            starts[i] = CPFactory.start(interval);
            ends[i] = CPFactory.end(interval);
            length[i] = length(intervals[i]);
            resource = CPFactory.plus(resource, pulse(interval, 1, Math.min(data.resourceCapacity, data.sizes[i])));
            height[i] = resource.heightAtStart(intervals[i]);
        }

        for (int i = 0; i < data.nbTasks; i++) {
            // Precedence constraints
            for (int k : data.successors[i]) {
                cp.post(endBeforeStart(intervals[i], intervals[k]));
            }
            // Size constraints
            cp.post(eq(mul(length[i], height[i]), data.sizes[i]));
        }

        // Resource constraint:
        cp.post(le(resource, data.resourceCapacity));

        // Objective
        CPIntVar makespan = max(ends);
        Objective obj = cp.minimize(makespan);

        // Search:
        DFSearch dfs = CPFactory.makeDfs(cp, and(firstFail(starts), firstFail(ends)));

        // Solution management:
        dfs.onSolution(() -> {
            System.out.println("---- solution ----");
            System.out.println("heights:"+Arrays.toString(height));
            System.out.println("makespan: " + makespan);
        });

        //Launching search:
        long begin = System.currentTimeMillis();
        SearchStatistics stats = dfs.optimize(obj);
        System.out.println(stats);
        long time = (System.currentTimeMillis() - begin)/1000;
        System.out.println("time(s):" + time);

    }


    public static void main(String[] args) throws Exception{
        ShipLoadingInstance data = new ShipLoadingInstance("data/SHIP_LOADING/shipLoading1.txt");
        ShipLoading sl = new ShipLoading(data);
    }
}

/**
 * Ship Loading Problem instance.
 *
 * @author Roger Kameugne
 */
class ShipLoadingInstance {
    public int nbTasks;
    public int nbResources;
    public int resourceCapacity;
    public int[] sizes;
    public ArrayList<Integer>[] successors;
    public int horizon;
    public String name;
    int sumSizes;

    public ShipLoadingInstance (String fileName) throws Exception {
        Scanner s = new Scanner(new File(fileName)).useDelimiter("\\s+");
        while (!s.hasNextInt()) s.nextLine();
        nbTasks = s.nextInt();
        nbResources = s.nextInt();
        resourceCapacity = s.nextInt();
        sizes = new int[nbTasks];
        successors = new ArrayList[nbTasks];
        sumSizes = 0;
        for (int i = 0; i < nbTasks; i++) {
            successors[i] = new ArrayList<>();
            sizes[i] = s.nextInt();
            sumSizes += sizes[i];
            int nbSucc = s.nextInt();
            if (nbSucc > 0) {
                for (int j = 0; j < nbSucc; j++) {
                    int succ = s.nextInt();
                    successors[i].add(succ - 1);
                }
            }
        }
        name = fileName;
        horizon = sumSizes;
        s.close();
    }
}