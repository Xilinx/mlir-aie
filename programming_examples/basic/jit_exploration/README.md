### Notebook

#### Use the Notebook
* Start a jupyter server at the root directory of your clone of `mlir-aie`.
  Make sure you use a terminal that has run the `utils/setup_env.sh` script
  so that the correct environment variables are percolated to jupyter.
  Below is an example of how to start a jupyter server:
  ```bash
  python3 -m jupyter notebook --no-browser --port=8080
  ```
* In your browser, navigate to the URL (which includes a token) which is found
  in the output of the above command.
* Navigate to `programming_examples/basic/tiling_exploration/introduction`
* Double click `taplib.ipynb` to start the notebook; choose the ipykernel called `ironenv`.
* You should now be good to go!

#### Run the Notebook as a Script
```bash
make clean
make run
```