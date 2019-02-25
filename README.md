# ATDA
This repository contains code to reproduce results from the paper:

**Improving the Generalization of Adversarial Training with Domain Adaptation （ICLR 2019）**

openreview report: https://openreview.net/forum?id=SyfIfnC5Ym

###### REQUIREMENTS

The code was tested with Python 3.6.5, Tensorflow 1.8.0, Keras 2.1.2, Keras_contrib 0.0.2,  Torchvision 0.2.1 and Numpy 1.14.3. 

###### EXPERIMENTS

We use  Adversarial Training (on FGSM)  with Domain Adaptaion to train a main model (modelZ)  for CIFAR-10 (default). 

```
python -m train_atda models/modelZ_atda --type=0
```

In addition, we use  Adversarial Training (on the noisy PGD)  with Domain Adaptaion to train a main model (modelZ)  for CIFAR-10 (default). 

```
python -m train_atda_npgd models/modelZ_atda --type=0
```

Then, we use Normal Training to train a model (modelC) for CIFAR-10 (default).

```
python -m train models/modelC --type=3
```

To use Original/ Standard Adversarial Training to train a main model：

```
python -m train_adv models/modelZ_adv --type=0
```

To use Ensemble Adversarial Training to train a main model：

```
# First train pre-trained models:
python -m train models/modelA --type=1
python -m train models/modelB --type=2
# use Ensemble Adversarial Training method to train with pre-trained models
python -m train_adv models/modelZ_ens models/modelA models/modelB --type=0
```

The accuracy of the models on the Fashion MNIST test set can be computed using:

```
python -m simple_eval test [model(s)]
```

To evaluate robustness to various attacks, we use:

```
python -m simple_eval [attack] [source_model] [target_model(s)] [--parameters (opt)]
```

The attack can be:

| Attack | Description               | Parameters                                                   |
| ------ | ------------------------- | ------------------------------------------------------------ |
| fgs    | Standard FGSM             | *eps* (the norm of the perturbation)                         |
| rfgs   | RAND+FGSM                 | *eps* (the norm of the total perturbation); *alpha* (the norm of the random perturbation) |
| pgd    | The iterative FGSM        | *eps* (the norm of the perturbation); *steps* (the number of iterative FGSM steps); alpha = eps/10.0 |
| mim    | Momentum Iterative Method | The parameter is fixed in the function *momentum_fgs* of the *fgs.py*. |

###### Acknowledgments

Code refer heavily to:  [Ensemble Adversarial Training](https://github.com/cxmscb/ensemble-adv-training) 
