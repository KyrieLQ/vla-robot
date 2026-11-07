#!/usr/bin/env python3
"""Toy checks of a conditional decision bound; not robotics performance evidence."""
import json, pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
    rng=np.random.default_rng(20261013);epsilon=.05;n=100000
    actual=rng.uniform(0,1,(n,3));estimate=actual+rng.uniform(-epsilon,epsilon,(n,3))
    best=estimate[:,1:].argmax(axis=1)+1;index=np.arange(n)
    estimated_gain=estimate[index,best]-estimate[:,0]
    switch=estimated_gain>2*epsilon
    actual_gain=actual[index,best]-actual[:,0]
    assert np.all(actual_gain[switch]>0)
    example={'true_continue':.80,'true_recover':.79,'estimated_continue':.75,'estimated_recover':.84,'epsilon':epsilon}
    assert example['estimated_recover']-example['estimated_continue']>0
    assert example['true_recover']-example['true_continue']<0
    assert example['estimated_recover']-example['estimated_continue']<2*epsilon
    result={'status':'PASS','scope':'Algebraic sanity check under a supplied uniform error bound only; no VLA experiment',
      'assumption':'All candidate expected net utilities share the same continuation/budget definition and have simultaneous absolute estimation error <= epsilon',
      'bound':'J_m-J_0 >= (Jhat_m-Jhat_0)-2*epsilon; switching only above 2*epsilon avoids a negative expected single-decision gain under the assumption',
      'toy_cases':n,'epsilon':epsilon,'bound_violations':int(np.sum(actual_gain[switch]<0)),
      'point_estimate_counterexample':example,
      'limitations':['Empirical ensemble standard deviation is not a proven uniform error bound.','A finite history may not be a sufficient state.','Changing the continuation policy or allowing repeated interventions invalidates a one-intervention comparison unless values are redefined.','No safety, convergence or task-success guarantee has been established for the proposed deployed system.']}
    out=ROOT/'reports/research_diagnosis_20261013/proposal_math_check.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
