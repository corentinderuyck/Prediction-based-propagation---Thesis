/*
 * MaxiCP is under MIT License
 * Copyright (c)  2023 UCLouvain
 */


package org.maxicp.cp.engine.constraints.setvar;

import org.maxicp.cp.engine.core.AbstractCPConstraint;
import org.maxicp.cp.engine.core.CPBoolVar;
import org.maxicp.cp.engine.core.CPSetVar;
import org.maxicp.cp.engine.core.CPSetVarImpl;

/**
 * Constraint that enforces a boolean variable to be true if a value is included in a set.
 */
public class IsIncluded extends AbstractCPConstraint {

    CPSetVar set;
    int v;
    CPBoolVar bool;

    /**
     * Creates a constraint that links a boolean variable to the inclusion of a value in a set.
     *
     * @param bool the boolean variable
     * @param set  the set variable
     * @param v    the value to be included/excluded
     */
    public IsIncluded(CPBoolVar bool, CPSetVar set, int v) {
        super(set.getSolver());
        this.set = set;
        this.v = v;
        this.bool = bool;
    }

    @Override
    public void post() {
        set.propagateOnDomainChange(this);
        bool.propagateOnFix(this);
        propagate();
    }

    @Override
    public void propagate() {
        if (bool.isTrue()) {
            set.include(v);
        } else if (bool.isFalse()) {
            set.exclude(v);
        }
        if (set.isIncluded(v)) {
            bool.fix(true);
        } else if (set.isExcluded(v)) {
            bool.fix(false);
        }
    }
}