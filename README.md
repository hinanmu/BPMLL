# BPMLL
## Dataset
[http://mulan.sourceforge.net/datasets-mlc.html][1]

### yeast
|name | domain | instances |nominal	|numeric|labels|cardinality	|density|distinct|
| ------ | ------ | ------ |------ |------ |------ |------ |------ |------ |
| yeast| biology | 2417	 |0|103	|14|4.237|0.303	|198|

## Evaluation
|evaluation criterion |BPMLL |
| ------ | ------ | 
| hamming loss| 0.247546346782988 |

## Requrements
- Python 3.6
- numpy 1.13.3
- tensorflow 1.10.0
- scikit-learn 0.19.1

## Parameter
- hidden_unit:0.8 * feature number
- Regularization alpha:0.1
- learning rate:0.05
- trainning step:500 * batch number

## Reference
[M.-L. Zhang, Z.-H. Zhou.Multilabel neural networks with applications to functional genomics and text categorization IEEE T. Knowl. Data En., 18 (10) (2006), pp. 1338-1351][2]


  [1]: http://mulan.sourceforge.net/datasets-mlc.html
  [2]: https://ieeexplore.ieee.org/abstract/document/1683770





