# shakespeare-gpt

## Overview
Welcome to my Deep learning project! This project is designed to create a language model similar to GPT 3, specifically trained on the first 1 million characters of Shakespeare, hence the name. The goal is to generate text that mimics Shakespeare's unique style. This project I walk through the steps of creating multiple models, each one iterating on the first, to reduce loss and improve model output. This project doesn't focus on any of the fine tuning stages. On each pass with each following model and additions, our output becomes better and better. We start with a simple bigram model and make our way to a transformer. This project showcases skills in deep learning, natural language processing, and PyTorch.

## Project Structure
The project is structured as follows:
- **models/bigram.py**: A simpler bigram model without token context. Loss: ~2.5
- **models/gpt.py**: A more complex GPT-like model with around 10M parameters. Includes multi-headed self-attention blocks, residual connections, dropout layers, layer norms, positional embeddings, and more. ~1
- **main.ipynb**: Jupiter Notebook containing the code for each model and steps broken down into bite sized chuncks.

## Features
- **Bigram Model**: A foundational model that predicts the next character based on the previous one, without taking the preceding tokens into context.
- **GPT Model**: An advanced model that uses all previous tokens in the context window (adjustable) to predict the next character.
- **Character-level Tokenization**: A simple and effective approach to tokenizing text where each character in the vocab is represented by a unique number.

## Installation
To run this project locally, you will need to have Python and PyTorch installed. You can install the necessary dependencies using the following command:
```sh
pip install -r requirements.txt
```

## Usage
### Training the Models and Generating Text
To train and run the models, navigate to the `model` directory and run the training script for the respective model you wish to generate:
```sh
python bigram.py
```

```sh
python gpt.py
```

## Results
The trained models can generate text that  resembles the style of Shakespeare. Here are some examples of generated text:

**Bigram Model V1 Output:**
```
od nos CAy go ghanoray t, co haringoudrou clethe k,LARof fr werar,
Is fa!


Thilemel cia h hmboomyorarifrcitheviPO, tle dst f qur'dig t cof boddo y t o ar pileas h mo wierl t,
S:
STENENEat I athe thounomy tinrent distesisanimald 3I: eliento ald, avaviconofrisist me Busarend un'soto vat s k,
SBRI he the f wendleindd t acoe ts ansu, thy ppr h.QULY:
KIIsqu pr odEd ch,
APrnes ouse bll owhored miner t ooon'stoume bupromo! fifoveghind hiarnge s.
MI aswimy or m, wardd tw'To tee abifewoetsphin sed The...
```


**GPT Model V1 Output:**
```
Wes le isen.
Woto teven INGO, ous into CYedd shou maithe ert thethens the the del ede cksy ow? Wlouby aicecat tisall wor
G'imemonou mar ee hacreancad hontrt had wousk ucavere.

Baraghe lfousto beme,
S m; ten gh;
S:
Ano ice de bay alysathef beatireplim serbeais I fard
Sy,
Me hallil:
DWAR: us,
Wte hse aecathate, parrise in hr'd pat
ERY:
Bf bul walde betl'ts I yshore grest atre ciak aloo; wo fart hets atl.

That at Wh kear ben.
 hend.
```


**GPT Model V2 Output:**
```
Bay.
A thill shat cof he hot mus finut twell ad I four shat son yald hat lods guks of som!

CSins
Gat son?-- dand un.

DAYOAUOMEN:
I dealll crefor hre bry epale gaeew are:
Thath, leanven-tford.

METI:
Fran thacepte, digns cot, Cow,
An pped lorcill air woth' that to thred? I tlill!

BUTA:
Mrows,
What loun be ping pe, aal ry fat, he cwor tend the to tis argito-, netoll bratens? I whit me flot nentes bre; manuth frifperen the hear
```

**GPT Model Final Version Output:**
```
O, that this too too solid flesh would melt,
Thaw and resolve itself into a dew!
Or that the Everlasting had not fix'd
His canon 'gainst self-slaughter! O God! God!
```

## Future Work and Improvement Considerations
- **Enhance the Model**: Explore more complex architectures and training techniques to improve the quality of generated text.
- **Expand the Dataset**: Train the model on other literary works to create models capable of generating diverse styles and enhancing model context and knowledge.
- **Fine-Tuning**: Experiment with fine-tuning the model on specific subsets of Shakespeare's works to generate genre-specific text (e.g., comedies, tragedies).

## Contact
For any questions or suggestions, please feel free to contact me at [owenhochwald@gmail.com].

Thank you for visiting my project! I hope you find it interesting and informative.

## References
- [Shakespeare's Works](https://www.gutenberg.org/files/100/100-h/100-h.htm)
- [PyTorch](https://pytorch.org/)
- [Attention Is All You Need Paper](https://arxiv.org/abs/1706.03762)

