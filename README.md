# TML_Assignment3
# Reproduce the Best Leaderboard Result

1. Open `Assignment3_notebook.ipynb` in Kaggle.
2. Enable a GPU accelerator.
3. Add the assignment dataset as a Kaggle input so that `train.npz` is available under `/kaggle/input`.
4. Run every notebook cell from top to bottom with the default configuration.
5. The notebook saves the model checkpoint to:

```text
/kaggle/working/model.pt
```

6. Run the notebook validation cell before submission. It checks that `model.pt` is a PyTorch state dict, loads into the selected ResNet architecture, and produces output shape `(1, 9)` for an input of shape `(1, 3, 32, 32)`.
7. To submit, store the API key as a Kaggle secret named `TML_API_KEY`, enable submission in the final cell, and run that cell once.
