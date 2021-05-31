"""
Input: container_path, data_dir, output_dir, timeout
Output: predictions in output_dir (or maybe metrics)
Steps:
1) download container into container_path (default = False)
2) Prepare container runscript
3) [GPU] Run container on toy cases (in data_dir)
4) [optional] Compute metrics on output predictions
5) Generate report with runtime and prediction results
6) Cleanup
"""
