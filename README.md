# 텐서플로와 머신러닝으로 시작하는 자연어 처리
*텐서플로와 머신러닝으로 시작하는 자연어처리* 8주차 강의 자료, TensorFlow 2.0 기반의 모델 구현 코드.



###  Table of Contents
- [Week1 - Basic TensorFlow](#week1---basic-tensorflow)
- [Week2 - Word Representation](#week2---word-representation)
- [Week3 - Text Classification](#week3---text-classification)
- [Week4 - Text Similarity](#week4---text-similarity)
- [Week5 - Text Generation](#week5---text-generation)
- [Week6 - Attention Mechanism](#week6---attention-mechanism)
- [Week7 - Transformer](#week7---transformer)
- [Week8 - BERT](#week8---bert)


## Week1 - Basic TensorFlow

###  Contents
- Estimator / Session
- tf.data / Placeholder & feed_dict
- TensorFlow Modules
- TensorFlow 2.0


###  Lecture note
- Basic TensorFlow [(Link)](https://github.com/reniew/learning_spoons_nlp/blob/master/lecture_note/)

###  Code
- Colab Notebook Setting [(Link)](https://github.com/reniew/learning_spoons_nlp/blob/master/code/Day1_1_Colab_Notebook_Setting.ipynb)
- Using TensorFlow Placeholder [(Link)](https://github.com/reniew/learning_spoons_nlp/blob/master/code/Day1_2_placeholder.ipynb)
- Using TensorFlow tf.data [(Link)](https://github.com/reniew/learning_spoons_nlp/blob/master/code/Day1_3_tf_data.ipynb)
- Using TensorFlow Session [(Link)](https://github.com/reniew/learning_spoons_nlp/blob/master/code/Day1_4_Session.ipynb)
- Using TensorFlow Estimator [(Link)](https://github.com/reniew/learning_spoons_nlp/blob/master/code/Day1_5_Estimator.ipynb)
- Using TensorFlow 2.0 [(Link)](https://github.com/reniew/learning_spoons_nlp/blob/master/code/Day1_6_Tensorflow2.0.ipynb)

## Week2 - Word Representation

###  Contents
  - Data Format(Tabular data, Image data, Text data)
  - One-Hot Representation
  - Bag of Words Representation
  - Neural Text Representation(Word2Vec, GloVe, FastText)

### Lecture note
- Word Representation [(link)](https://github.com/reniew/learning_spoons_nlp/blob/master/lecture_note/Lecture2.pptx)

### Code
- One Hot Encoding Representation [(link)](https://github.com/reniew/learning_spoons_nlp/blob/master/code/Day2_1_One_Hot_Encoding.ipynb)
- Bag of Words Representation [(link)](https://github.com/reniew/learning_spoons_nlp/blob/master/code/Day2_2_Bag_Of_Words.ipynb)
- Word2Vec(CBOW) [(link)](https://github.com/reniew/learning_spoons_nlp/blob/master/code/Day2_3_Word2Vec(CBOW).ipynb)
- Word2Vec(Skip-Gram) [(link)](https://github.com/reniew/learning_spoons_nlp/blob/master/code/Day2_4_Word2Vec(skip_gram).ipynb)

## Week3 - Text Classification

### Contents
- Text Classification
- Text Classification Dataset
- Classification Evaluate
- Natural Language Inference
- Convolutional Neural Networks for Sentence Classification
- Character-level Convolutional Neural Networks for Text Classification

### Lecture note
- Text Classification [(link)](https://github.com/reniew/learning_spoons_nlp/blob/master/lecture_note/Lecture3.pptx)

### Code
- Naver Movie Review Corpus EDA & Preprocessing [(link)](https://github.com/reniew/learning_spoons_nlp/blob/master/code/Day3_1_Naver_Movie_Review_EDA_Preprocessing.ipynb)
- Implementation of Concolutional Neural Networks for Sentence Classification  [(link)](https://github.com/reniew/learning_spoons_nlp/blob/master/code/Day3_2_Yoon_Kim_Model.ipynb)
- Implementation of RandomForest Model [(link)](https://github.com/reniew/learning_spoons_nlp/blob/master/code/Day3_3_RandomForest.ipynb)

## Week4 - Text Similarity

### Contents
- Text Similarity
- Text Similarity Dataset
- Measuring Similarity(Jacard, Cosine, Euclidean Manhattan)
- Siamese Recurrent Architectures for Learning Sentence Similarity(MaLSTM)
- Tree Based Algorithm (XGBoost, RandomForest, Bagging, Boosting)

### Lecture note
- Text Similarity [(link)](https://github.com/reniew/learning_spoons_nlp/blob/master/lecture_note/Lecture4.pptx)
- Tree Based Algorithm [(link)](https://github.com/reniew/learning_spoons_nlp/blob/master/lecture_note/Tree_based.pptx)

### Code
- Quora Question Paris EDA & Preprocessing [(link)](https://github.com/reniew/learning_spoons_nlp/blob/master/code/Day4_1_Text_Similarity_EDA_Preprocessin.ipynb)
- Implementation of MaLSTM [(link)](https://github.com/reniew/learning_spoons_nlp/blob/master/code/Day4_2_Ma_LSTM.ipynb)
- Implementation of XGBoost Model [(link)](https://github.com/reniew/learning_spoons_nlp/blob/master/code/Day4_3_XGBoost.ipynb)

## Week5 - Text Generation

### Contents
- Text Generation
- Neural Machine Translation
- Neural Machine Translation Dataset
- Machine Translation Evaluate(BLEU Score)
- Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation
- Sequence to Sequence Learning with Neural Networks

### Lecture note
- Text Generation [(link)](https://github.com/reniew/learning_spoons_nlp/blob/master/lecture_note/Lecture5.pptx)

### Code
- Chatbot Data for Korean EDA & Preprocessing [(link)](https://github.com/reniew/learning_spoons_nlp/blob/master/code/Day5_1_EDA.ipynb)
- Implementation of Sequence to Sequence [(link)](https://github.com/reniew/learning_spoons_nlp/blob/master/code/Day5_2_seq2seq_Model.ipynb)

## Week6 - Attention Mechanism

### Contents
- Attention Mechanism
- Neural Machine Translation by Jointly Learning to Align and Translate
- Various Attention Mechanism
- Additional Technique for Neural Machine Translation

### Lecture note
- Attention Mechanism [(link)](https://github.com/reniew/learning_spoons_nlp/blob/master/lecture_note/Lecture6.pptx)

### Code
- Implementation of Sequence to Sequence with Attention [(link)](https://github.com/reniew/learning_spoons_nlp/blob/master/code/Day6_1_advanced_seq2seq_Model.ipynb)

## Week7 - Transformer

### Contents
- History of Neural Machine Translation
- Attention is All You Need

### Lecture note
- Transformer [(link)](https://github.com/reniew/learning_spoons_nlp/blob/master/lecture_note/Lecture7.pptx)

### Code
- Implementation of Transformer [(link)](https://github.com/reniew/learning_spoons_nlp/blob/master/code/Day7_1_Transformer.ipynb)

## Week8 - BERT

### Contents
- BERT: Pre-training of Deep Bidirectional Transformer for Language Understanding

### Lecture note
- BERT [(link)](https://github.com/reniew/learning_spoons_nlp/blob/master/lecture_note/Lecture8.pptx)

### Code
- Fine-Tuning Pre-Trained BERT to NSMC Task  [(link)](https://github.com/reniew/learning_spoons_nlp/blob/master/code/Day8_1_BERT_nsmc.ipynb)

