+++
title = 'How do Transformers Work?'
weight = 2
+++
Language models encode statistical information about language learned from large text corpora, often data scraped from the internet. With this information, they are able to predict, given a sequence of words, the next one. Roughly speaking, a sentence can be constructed by repeatedly predicting words in this manner.

Most modern language models used in generative tasks are autoregressive[^huyen2025]. These models predict the next input based on their own output, and can be leveraged in a prediction loop to form whole sentences. If I asked a GPT-style transformer "how are you?", behind the scenes, it might see something like this:[^context_note]

```txt
SYSTEM:
You are a happy, helpful chatbot that answers questions.

USER:
how are you?

ASSISTANT:
```

And so it fills in the part after "ASSISTANT":
```txt
"ASSISTANT:\n"
"ASSISTANT:\nI"
"ASSISTANT:\n I am"
"ASSISTANT:\n I am great"
"ASSISTANT:\n I am great!"
"ASSISTANT:\n I am great!<EOS>"
```

Bearing in mind that with every new prediction it's also using all the context before it starting from "SYSTEM:". That is, let's say we used 15 tokens for the first message—15 tokens are passed into the model to get the 16th, then 16 tokens are passed in to get the 17th, and so on. The `<EOS>` token is a special "end of sequence" token instructing the surrounding program to stop the loop and return the answer.[^KV]

In reality it predicts the next token in its vocabulary, not the next English word per se. And what even are tokens?

## Tokens and Tokenization

The smallest unit of language in such models is the token. Tokens can be smaller or larger than words—it depends on the choice of vocabulary.

-- IMAGE OF TOKENIZATION HERE --

The vocabulary in this sense is the set of all tokens a model uses. When we call a model in code, using a high-level API such as an LLM library or a provider SDK (e.g., the anthropic library in Python), what happens is that the passed-in string is split into a list of tokens by a tokenizer (locally or by the provider depending on the tools used). This process is called tokenization. A tokenizer is itself a model, typically learned from word-fragment frequencies in a large body of text.

> [!NOTE]
> Why tokens? If you had trained a model to only work on words on a vocabulary that were really just the English lexicon, you encounter the problem that unseen words cannot be interpreted by the model. A compound word is a good example, like "housewarming"—"house" and "warming" may have been encountered, yet "housewarming" would be uninterpretable. The model would be forced to shoulder a giant vocabulary, which, as we shall see, takes up computational resources. Modern tokenizers also start from 256 byte values, so no string is left out of the vocabulary.
> 
> However, as the image above shows, some tokens can be quite long—entire words in fact, possibly several. This makes the encoding very efficient, as an entire word can be packed into a single token.
> 
> Now, on the opposite end, why not individual characters? You'd only need a vocabulary of ~100 tokens, right? While true, this would be a vastly inefficient use of resources as now the model would process unnecessarily long strings of tokens.

[^huyen2025]: Huyen, Chip. AI Engineering: Building Applications with Foundation Models. O'Reilly Media, 2025.

[^context_note]: This block of information is also called the "context", representing the whole list of tokens that a model sees. Context engineering, writing a context that makes the model reliable, secure, and feature-rich, is an active field in and of itself. In contrast, the user- and system-specific blocks within the context shown are called the user and system prompt respectively, and "prompt engineering" as it is commonly phrased largely concerns those two blocks.

[^KV]: TODO add footnote