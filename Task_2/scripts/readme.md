# Scripts for task 2 submissions

Short description of scripts:
- `test_container.py`: Test your container locally (run and measure time). This is similar to what is done in the test runs of the validation phase.
- `run_submission.py`: Used by `test_container.py` and in the actual evaluation system to execute the singularity container.
- `metric_evaluation.py`: Used by `test_container.py` to compute metrics. Note that this implementation based on medpy is not identical to the one used during the testing phase.
