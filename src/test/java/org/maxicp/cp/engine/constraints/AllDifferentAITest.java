/*
 * MaxiCP is under MIT License
 * Copyright (c)  2023 UCLouvain
 */

package org.maxicp.cp.engine.constraints;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.maxicp.cp.engine.CPSolverTest;
import org.maxicp.cp.engine.core.CPIntVar;
import org.maxicp.cp.engine.core.CPSolver;
import org.maxicp.cp.CPFactory;

import java.util.Arrays;
import java.util.HashSet;

import static org.junit.jupiter.api.Assertions.*;


public class AllDifferentAITest extends CPSolverTest {

    @ParameterizedTest
    @MethodSource("getSolver")
    public void allDifferentTest1(CPSolver cp) {

        CPIntVar[] x = CPFactory.makeIntVarArray(cp, 5, 5);

        cp.post(new AllDifferentAI(x));
        cp.post(CPFactory.eq(x[0], 0));
        for (int i = 1; i < x.length; i++) {
            assertEquals(4, x[i].size());
            assertEquals(1, x[i].min());
        }
    }


    private static CPIntVar makeIVar(CPSolver cp, Integer... values) {
        return CPFactory.makeIntVar(cp, new HashSet<>(Arrays.asList(values)));
    }


    @ParameterizedTest
    @MethodSource("getSolver")
    public void allDifferentTest2(CPSolver cp) {
        CPIntVar[] x = new CPIntVar[]{
                makeIVar(cp, 1),
                makeIVar(cp, 1, 2),
                makeIVar(cp, 1, 2, 3)};

        cp.post(new AllDifferentAI(x));

        assertEquals(x[1].min(), 2);
        assertEquals(x[1].size(), 1);
        assertEquals(x[2].min(), 3);
        assertEquals(x[2].size(), 1);
    }

    @ParameterizedTest
    @MethodSource("getSolver")
    public void allDifferentTest3(CPSolver cp) {
        CPIntVar[] x = new CPIntVar[]{
                makeIVar(cp, 1, 2, 3),
                makeIVar(cp, 1, 2),
                makeIVar(cp, 1)};

        cp.post(new AllDifferentAI(x));

        assertEquals(x[0].min(), 3);
        assertEquals(x[0].size(), 1);
        assertEquals(x[1].min(), 2);
        assertEquals(x[1].size(), 1);
    }

    @ParameterizedTest
    @MethodSource("getSolver")
    public void allDifferentTest4(CPSolver cp) {
        CPIntVar[] x = new CPIntVar[]{
                makeIVar(cp, 1),
                makeIVar(cp, 1, 2, 3),
                makeIVar(cp, 1, 2, 3)};

        AllDifferentAI allDiff = new AllDifferentAI(x);

        cp.post(allDiff);

        // check swapping
        assertEquals(allDiff.nbNonFixed.value(), 2);
        assertEquals(allDiff.nonfixed[0], 2);
        assertEquals(allDiff.nonfixed[1], 1);
        assertEquals(allDiff.nonfixed[2], 0);
    }

    @ParameterizedTest
    @MethodSource("getSolver")
    public void allDifferentTest5(CPSolver cp) {
        CPIntVar[] x = new CPIntVar[]{
                makeIVar(cp, 1),
                makeIVar(cp, 1, 2),
                makeIVar(cp, 1, 2, 3)};

        AllDifferentAI allDiff = new AllDifferentAI(x);

        cp.post(allDiff);

        // check swapping
        assertEquals(allDiff.nbNonFixed.value(), 0);
        assertEquals(allDiff.nonfixed[0], 2);
        assertEquals(allDiff.nonfixed[1], 1);
        assertEquals(allDiff.nonfixed[2], 0);
    }

    @ParameterizedTest
    @MethodSource("getSolver")
    public void allDifferentTest6(CPSolver cp) {
        CPIntVar[] x = new CPIntVar[]{
                makeIVar(cp, 1, 2),
                makeIVar(cp, 1),
                makeIVar(cp, 1, 2, 3)};

        AllDifferentAI allDiff = new AllDifferentAI(x);

        cp.post(allDiff);

        // check swapping
        assertEquals(allDiff.nbNonFixed.value(), 0);
        assertEquals(allDiff.nonfixed[0], 2);
        assertEquals(allDiff.nonfixed[1], 0);
        assertEquals(allDiff.nonfixed[2], 1);
    }


}
