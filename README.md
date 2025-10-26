# Thesis code — README

## Files

* `filtered_xml_instances_test`: XML instances (CSP models) for testing.
* `filtered_xml_instances_train`: XML instances (CSP models) for training the AI model.
* `python`: Python source files (see next section for details).
* `src`: MaxiCP solver source code, modified to be compatible with the Python code.
* `instances_test.csv` / `instances_train.csv`: CSV files with statistics (number of solutions, number of nodes, failures, and execution time) for each problem in the corresponding folder (test / train) when solved with the baseline MaxiCP solver (i.e., **without** predictions from the AI model).
* `submit.sh`: A file to submit jobs to a SLURM cluster.

---

## Changes made to MaxiCP

In `src/main/java/org/maxicp/cp/engine/constraints` there is a new constraint file: `AllDifferentAI.java`.
This implements an `AllDifferent` constraint that delegates propagation to the Python AI model. It uses a socket to transfer variable identifiers and domains to the Python process, lets Python perform propagation, and receives the updated domains back.

Also in `src/main/java/org/maxicp/cp/engine/constraints` is `AllDifferentDC.java`. This is the default `AllDifferent` constraint used by the solver. Its code contains a boolean flag `USE_AI_MODEL` that selects whether the solver uses the classical Régin algorithm implementation (`AllDifferentDC`) or the AI-driven propagation (`AllDifferentAI`).

Under `src/main/java/org/maxicp` there is a `RunXSCP3` folder containing a Java entry-point file. This provides several utilities:

* `runAllInstanceFolderAI`: starts (and later closes) the socket that links the Java code to the Python model, then runs every XML instance in the folder given as argument with different thresholds, and saves results to a CSV file. To use the AI model during these runs, set `USE_AI_MODEL` in `AllDifferentDC.java` to `true`.

* `runAllInstances`: solves the XML instances in a folder and has two main purposes: (1) collect baseline solver statistics (classic solver) and (2) collect training data for the model (variable domains before and after propagation). To save training data, set the boolean `saveState` in `AtLeastNValueDC.java` to `true`. The `runAllInstances` function accepts an argument `oneFile` that controls whether all data are written into a single CSV file or into separate files per instance. The `folderName` variable in `AtLeastNValueDC.java` will be used as the folder name for saved data. Using a single file is more convenient for training the model and a separate files per instance are more convenient for evaluating the trained model per instance.

---

## Python

This folder contains the following files/directories:

* `process_data.py`: script to preprocess the training data (see the thesis/report for details).
* `plots`: code and generated plots used in the report, based on the contents of `data`.
* `data`: folder containing datasets, results, and trained AI models.
* `machine_learning`: code to build (`build_...`), test (`test_...`) and integrate with Java (`use_...`) the different models. The `others` subfolder contains alternative model architectures that were left unfinished.

---

## Run

The expected environment layout to run the Java solver together with the Python AI model is:

* **python**: `use_model_in_java.py`, `env` (Python environment with required packages)
* **data**: `model.pth`
* **java**: `maxicp.jar`, `filtered_xml_instances_test`, `filtered_xml_instances_train`

**Directory layout (project root)**

```
project-root/
├─ python/
│  ├─ use_model_...py
│  └─ env/                 # Python virtual environment with required packages
├─ data/
│  └─ model.pth            # trained model used by the Python server
└─ java/
   ├─ maxicp.jar
   ├─ filtered_xml_instances_train/
   └─ filtered_xml_instances_test/
```
