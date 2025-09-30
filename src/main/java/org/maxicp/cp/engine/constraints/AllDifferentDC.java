/*
 * MaxiCP is under MIT License
 * Copyright (c)  2023 UCLouvain
 */

package org.maxicp.cp.engine.constraints;

import org.maxicp.cp.engine.core.AbstractCPConstraint;
import org.maxicp.cp.engine.core.CPIntVar;
import org.maxicp.cp.engine.core.CPIntVarImpl;
import org.maxicp.util.GraphUtil;
import org.maxicp.util.GraphUtil.Graph;
import org.maxicp.util.exception.InconsistencyException;

import java.util.ArrayList;
import java.util.Arrays;
/**
 * Arc Consistent AllDifferent Constraint
 *
 * Algorithm described in
 * "A filtering algorithm for constraints of difference in CSPs" J-C. Régin, AAAI-94
 */


public class AllDifferentDC extends AbstractCPConstraint {

    // Choose the implementation based on a flag
    private static final boolean USE_AI_MODEL = true;

    private final AbstractCPConstraint impl;

    public AllDifferentDC(CPIntVar... x) {
        super(x[0].getSolver());

        if (USE_AI_MODEL) {
            impl = new AllDifferentAI(x);
        } else {
            impl = new AtLeastNValueDC(x, new CPIntVarImpl(x[0].getSolver(), x.length, x.length));
        }
    }

    @Override
    public void post() {
        impl.post();
    }

    @Override
    public void propagate() {
        impl.propagate();
    }
}