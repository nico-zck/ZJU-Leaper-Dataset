### API of ZJU-Leaper

Please use `ZLFabric` int the `zl_fabric.py` to access ZJU-leaper dataset.

```python
class ZLFabric:
    def __init__(self, dir: str, fabric: Union[str, int], setting: Union[str, int], seed: int = None):
        """
        Create an object to manage ZJU-Leaper dataset.
        
        :param dir: the base directory for the ZJU-Leaper dataset.
        :param fabric: int or string to specify using "patternX" or "groupX".
        :param setting:
            |Available strings for setting parameter:
            |   "setting1": Normal sample only (annotation-free);
            |   "setting2": Small amount of defect data (with mask annotation);
            |   "setting3": Large amount of defect data (with label annotation);
            |   "setting4": Large amount of defect data (with bounding-box annotation);
            |   "setting5": Large amount of defect data (with mask annotation);
        :param seed: random seed.
        """

    def prepare_train(self) -> Tuple[ZLIMGS, ZLIMGS, ZLIMGS, ZLIMGS]:
        """
        Preparing ZLImages from training set.
        :return: [Normal ZLImages, Defective ZLImages]
        """

    def prepare_k_fold(self, k_fold: int, shuffle: bool = True) \
            -> List[Tuple[ZLIMGS, ZLIMGS, ZLIMGS, ZLIMGS, ZLIMGS, ZLIMGS]]:
        """
        Preparing ZLImages from cross validation.
        :return: [(zlimgs_k_train_normal, zlimgs_k_train_defect, zlimgs_k_dev_normal, zlimgs_k_dev_defect)]
        """

    def prepare_test(self) -> Tuple[ZLIMGS, ZLIMGS]:
        """
        Preparing ZLImages from test set.
        :return: [Normal ZLImages, Defective ZLImages]
        """
```

### API of evaluation

Please use `ZLEval` in the `zl_eval.py` to evaluate an inspection algorithm.

```python
class ZLEval:
    def __init__(self, binary_target: np.ndarray, binary_pred: np.ndarray) -> None:
        """
        Create an object to evaluate an inspection algorithm.
        :param binary_target: binarized mask of ground truths. 
        :param binary_pred: binarized mask of prediction.
        """

    def evaluate(self) -> OrderedDict:
        """
        Calculate all metrics on the results and return metrics as a dict.
        :return:
        """

    def summarize(self):
        """
        Pretty printing metrics dict
        :return:
        """
```