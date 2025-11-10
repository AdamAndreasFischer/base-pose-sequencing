from tensorboard import program

tracking_address = "/home/adamfi/codes/base-pose-sequencing/runs"
"""Start tensorboard"""
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', tracking_address])
url = tb.launch()
print(f"Tensorflow listening on {url}")
""""""