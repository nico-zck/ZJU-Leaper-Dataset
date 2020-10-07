- Methods of `seg_one_class` try to solve problems of Setting1 with only normal samples.
- Methods of `seg_augment` and `seg_transfer` try to solve Setting2 with a few defects.
- Methods of `seg_weakly` try to solve Setting3 and Setting4 with only labels and bounding-boxes as annotations
  respectively.
- Methods of `seg_fully` use all data in the ZJU-Leaper to solve Setting5, which will display the best performance but
  not possible for practical applications.

For more detailed information, please refer to our paper.