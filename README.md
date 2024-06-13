# shakespeare-gpt

## Overview
Welcome to my Deep learning project! This project is designed to create a language model similar to GPT 2, specifically trained on the entire works of Shakespeare, hence the name. The goal is to generate text that mimics Shakespeare's unique style. This project showcases skills in deep learning, natural language processing, and PyTorch.

## Project Structure
The project is structured as follows:
- **models/bigram.py**: A simpler bigram model.
- **models/gpt.py**: A more complex GPT-like model.
- **main.ipynb**: Jupiter Notebook containing the code for each model and steps broken down into bite sized chuncks.

## Features
- **Bigram Model**: A foundational model that predicts the next character based on the previous one, without taking the preceding tokens into context.
- **GPT Model**: An advanced model that uses all previous tokens in the context window to predict the next character.
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

**Bigram Model Output:**
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

**GPT Model Output:**
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
