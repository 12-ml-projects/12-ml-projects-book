+++
title = 'Crafting a Language'
weight = 3
+++


## Information Theory, Cross-Entropy and Language Complexity
-- Maybe start here with a section on The Gold Bug by Poe or The Adventures of the Dancing Men by Conan Doyle hmm, substitution cipher trick looking at character frequencies --

Language modeling as a means to encode and use statistical information of language goes back at least as far as the 50's to early works by Claude Shannon. One paper, Prediction and Entropy of Printed English, uses the information theoretical machinery from his earlier landmark paper to explore the entropy of the English language. Entropy can be intuitively understood as a measure of the information capacity of a language; if English were encoded in the most efficient way to 1s and 0s, the entropy of English would be the average number of digits required per letter in the original English.

The entropy of English can be estimated through a game involving an examiner and participant; select an English passage unfamiliar to the participant and ask them to guess the first letter following the passage[^footenote_guessing]. If the guess is correct, they are told so, the letter is kept, and the game moves on to guess the next letter. If wrong, the participant will be told the correct letter and the game likewise proceeds. The result of a typical experiment is taken from Shannon's 1951 paper[^shannon1951].

```txt
THE ROOM WAS NOT VERY LIGHT A SMALL OBLONG
----ROO------NOT-V-----I------SM----OBL---
READING LAMP ON THE DESK SHED GLOW ON
REA----------O------D----SHED-GLO--O-
POLISHED WOOD BUT LESS ON THE SHABBY RED CARPET
P-L-S-----O---BU--L-S--O------SH-----RE--C-----
```
Here complete sentences show the original text, and dashes represent letters correctly guessed. The participant guessed above 70% of the letters correctly.

The below line, which Shannon calls the reduced text, seems to contain a lot less information. Surprisingly, both lines encode the same information in principle. That is to say, if we had an identical twin of the participant, they could, in principle reconstruct the original text from the reduced text. Here's why: suppose we had our original passage minus the continuation, and instead only had the reduced text from which to construct it. Our twin participant could play the same game as before. They would answer the same. We are only interested in the unknown dashed letters. And any letter the twin participant supplies in place of an unknown dashed letter would be correct, since the twins answer the same.

The game above measures the conditional entropy of one letter given the previous \(N\).

\[
F_{N} = H(X_{t} | X_{t-N},...,X_{t-1})    
\]









Variants of the game exist, involving bets or repeated guesses.

The reduced text can be thought of as the original text passed through an encoder.

There is a limit to how many characters can be removed.

The results of this experiment give bounds on the conditional entropy for a single letter of English text given knowledge of the previous N characters.




since they would respond in the same way when playing the same guessing game.





Here the human is playing a role analogous to the language models of today.






Entropy relates to the following question; how well can the next letter be predicted by the preceding \(N\) letters?

Take a language model. It predicts the next token from the list of previous tokens. How well can we predict the next token?

When we take such a language model, what do we want it to do? Predict the next token with absolute certainty, given the previous \(N\) tokens? Such a model is a function \(T_{t+1}=M(T_{t-N},...,T_{t})\). But this is not quite it; I can take an input like "the man is ", which can be continued in any number of ways. Recall that a forward pass as above is deterministic, so it would always predict the same sequence. Let us take the next best thing; a distribution over tokens.

So for each token \(T \in A\) in the alphabet, let the model assign a probability \(P(T=t | T_{1},...,T_{N}) = M(T_{1},...,T_{N})\) over all possible choices of \(T_{t \in \left\{ 1,...,N \right\}}\). Now what does it mean for the model to "do well"? How would we test it? We should expect the distribution over tokens \(P(T=t | T_{0}...T_{N})\) to match that of the underlying language.

And it's certainly possible to estimate this distribution for small context windows using n-grams. N-grams are short sequences of tokens. Trigrams, for example, could be `aal`, `zaw`, `cda` and so on. One can estimate the above probability by counting trigrams from the English language—from books, magazines, online forums, social media—and calculating the probabilities as

\[
P(T_{3} | T_{1}T_{2})=\frac{C(T_{1}T_{2}T_{3})}{C(T_{1}T_{2})}
\]
Where \(C(x)\) is the number of occurences of sequence \(x\) in the sample[^trigrams_note].

If we had an infinite, representative sample of English to construct these \(N\)-grams from, such probabilities would correspond to the true probabilities in the English language of the next token given a previous set of \(N\) tokens. One could in theory use this to test your model against. In practice, you can already compare outputs generated by the model against bigrams trained on small datasets alone, with the bigram model acting as a baseline. This is exactly what Karpathy did in his GPT tutorial. The deeper point is, you want to compare the model's predicted distribution against the "true" distribution underlying the language.

But given the true distribution \(P(T=t | T_{1},...,T_{N})\) and the model's estimate \(P_{M}(T=t | T_{1},...,T_{N})\), how do we mathematically compare the two?


-- Maybe here connect it to cross entropy loss, which tells us how many bits on average would it cost to encode the observed outcome using your model's predicted distribution --

-- Tie this to conditional entropy and showing that it's a loss floor for all models... hmmm --

-- Tie that into the context free grammar, for which we can estimate these quantities, and use that as a means of comparison, saying how good the model is --


## Context Free Grammars
Before we train GPT-1 on real data, we shall train it on a dummy, artificial language. But how do we construct an artificial language? One starts with a grammar. There are actually many "families" of grammars, some of which are more powerful than others. We will focus on context free grammars (CFGs)[^chomsky1956]. CFGs formalise a long-standing notion that sentences are made up of constituents, which in turn can be made up of other constituents. That is to say, language is hierarchical.

Our dummy language is defined by the following specification file.
```txt
S       -> NP_sg VP_sg Dot | NP_pl VP_pl Dot
NP      -> NP_sg | NP_pl
NP_sg   -> Det_sg Nom_sg | Det_sg Nom_sg Rel
NP_pl   -> Det_pl Nom_pl | Det_pl Nom_pl Rel
Rel     -> "that" NP_sg Vt_sg | "that" NP_pl Vt_pl
Nom_sg  -> Adj Nom_sg | N_sg
Nom_pl  -> Adj Nom_pl | N_pl
VP_sg   -> Vi_sg | Vt_sg NP
VP_pl   -> Vi_pl | Vt_pl NP
Dot     -> "."
Det_sg  -> "a" | "this" | "the"
Det_pl  -> "some" | "these" | "those"
N_sg    -> "dog" | "cat"
N_pl    -> "dogs" | "cats"
Adj     -> "small" | "old" | "happy" | "big"
Vi_sg   -> "sleeps" | "runs" | "sees" | "eats"
Vi_pl   -> "sleep" | "run" | "see" | "eat"
Vt_sg   -> "sees" | "chases"
Vt_pl   -> "see" | "chase"
```
A CFG has (1) a set of grammar rules or productions, which express the way symbols in the language can be decomposed and (2) a lexicon, or set of terminal symbols—the words of the language. In the language specification above, the names of symbols relate to their grammatical roles—"S" for sentence, "NP_sg" for singular noun phrase, and so on—these are the parts of speech (PoS) for the language. The **|** symbol defines alternative productions—when decomposing a PoS, you pick one production. Sentences can be constructed by sampling from this grammar. You can sample, for example, "the cat sleeps." as follows: you start with **S**, and decompose it as **NP_sg VP_sg Dot**. The **Dot** can be readily converted to the terminal symbol ".", and the **NP_sg** is decomposed as **Det Nom_sg** and so on. Below is a printed parse tree that shows the full deconstruction.

```txt
               S              
          _____|____________  
       NP_sg          |     | 
   ______|_____       |     | 
  |          Nom_sg VP_sg   | 
  |            |      |     | 
Det_sg        N_sg  Vi_sg  Dot
  |            |      |     | 
 the          cat   sleeps  . 
```

This also goes the other way—when you take a sentence like the above, there is a way to construct a parse tree from which the sentence came. In fact, sometimes there are several possible parse trees, in which case the grammar is said to be ambiguous. The grammar above is unambiguous.

This is all we need to get started. Following an artificial, formally-defined language has several benefits for us working to make an LLM. We can sample any number of sentences from this grammar, giving us unlimited data to train on. We can also grammar check the outputs of our language models—a sentence is grammatical if it has one or more possible parse trees. Thus we have a very simple check for correctness. But the main benefit, for the time being, is that we can keep the language small and simple, and by extension our model as well. We can train a tiny GPT model in a matter of minutes on such a language, and use it as a baseline for correctness. Scaling to a larger model—once we are certain the smaller model works—is as simple as swapping a few numbers in the Python code.













With this grammar we can construct our own treebank—a corpus in which every sentence is annotated with a parse tree.




We shall first build a small version of GPT-1 and test it on an artificial language.






The size and training time of a GPT model needed to reach a given benchmark scales with the complexity, or entropy, of the language it is trying to model as well as the amount of data available. For a complex language like English, it may take days on a consumer device to achieve modest results. For a subset of English this problem is much more tractable on a consumer device.

This is exactly what Andrej Karpathy, founding member of OpenAI, did in [https://www.youtube.com/watch?v=kCc8FmEb1nY](Let's Build GPT: from scratch, in code, spelled out) video, where he trains it on the Tiny Shakespeare[^Karpathy2020] dataset. He trains the model in ~15 minutes on an NVIDIA A100 GPU.

We will take a slightly different approach, constructing an artificial language from a context free grammar on which to train our model. There are several reasons for this approach:

- We can effortlessly pick the complexity, or entropy, of our language, and even estimate it mathematically
- We can benchmark our model formally against the grammar




We use NLTK[^nltk]

https://arxiv.org/pdf/2512.24969


[^ntlk]: Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.
[^footnote_guessing]: With spaces included, that is.
[^shannon1951]: Shannon, C. E. (1951). Prediction and next-letter prediction of printed English. Bell System Technical Journal, 30(1), 50–64. doi.org
[^trigrams_note]: The denominator can also be expressed as a sum over trigrams.
[^Karpathy2020]: Karpathy, A. (2015). tiny_shakespeare [Data set]. Hugging Face. https://huggingface.co/datasets/karpathy/tiny_shakespeare
[^chomsky1956]: Chomsky, N. (1956). Three models for the description of language. IRE Transactions on Information Theory, 2(3), 113–124. doi.org