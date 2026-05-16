Yaron
While *HuggingFaceTB/SmolLM2-360M-Instruct generate reasonable answers, both before and after fine tuning, google/flan-t5-small generates total garbage responses, both the base model and the fine tuned one (this is a typical response: ""A a aas a aas a aas a aas a aas a aas a aas a aas a aas...")
I have tried a couple of prompts, but could not get better results.
Has anyone got better experience with this model?

flan-t5 was pre-trained on short , structuted text to text tasks- translation, classification. asking him to generate multe sentence explanations at defferent expertise level is outside its training distribution

Has anyone seen this with QLoRA + FLAN‑T5? Is it acceptable for the assignment to use  plain LoRA (no 4‑bit) for FLAN‑T5, since that setup gives a normal loss? @Yael Hamrani
Ofer Goren — 1:53 PM
Yes, had the same issue. Fall back to LoRA. small datasets are not good for decoder-encoder architecture 

thanks, @Yael Hamrani . In the class we learned that usually we should use rank 1 or 2, but since here the models are smaller than usual, can we use higher ranks?
Yael Hamrani — Yesterday at 12:08 PM
Hi , yes you should experiment with higher ranks .
Ofer Goren — Yesterday at 11:15 PM
I used 8.

FLAN T5 was pretrained in bf16. 
bf16 and fp16 are both 16 bit, but the bit layout is different

Same total bits, different tensor dtype. T5's weights were tuned for bf16 range, so fp16 inference/training is fundamentally unstable for this family. 

Colab runbook
Runtime  Change runtime type  A100 or L4 GPU (T4 has no native bf16; it'll emulate and be slow, but won't NaN).
!pip install -q -U transformers peft bitsandbytes accelerate datasets
In BitsAndBytesConfig: bnb_4bit_compute_dtype=torch.bfloat16
In Seq2SeqTrainingArguments: bf16=True, fp16=False
Don't call .to("cuda") on the 4-bit model  device_map="auto" handles it.

Was driving me crazy till I found this, it will train but like other said, the model will still flop at inference, at least for me

After several attempts, I can definitely say that FLAN-T5 is very sensitive to the training config. The best results were achieved using LR=1e-4, rank=8, and target_modules=["q", "v"]
Despite the higher losses during training, I got answers that looked like speech rather than gibberish babbling. But still, they were mostly hallucinated answers.

ael Hamrani Could you please clarify whether we should do it manually or use automated evaluation (LLM-as-a-judge)?
Yael Hamrani — 4:09 PM
On the test set manually, it is only 20  short q&a examples.

for the submission, should we only upload the final notebook, or do you also want us to include the  the Excel file with our manual evaluation ?
Yael Hamrani — 7:18 AM
Show an examples in the notebook, print the df csv, give insights and discuss what you got.

Gennadii Borisov
 — 5/9/26, 3:31 PM
Hi. I didn't understand the point of the last task. If we assign embeddings as random numbers, then whatever sentence we take, all the maximal elements will be diagonal. Because we're simply flipping the vector onto itself. So there will be no difference between a cat and a dog. I tried using real trained embeddings, and at least some sense appeared.
Boris Kogan — 7:40 AM
@Yael Hamrani  Same question for multihead. If we multiply one random matrix with 3 times with random matices  to get Q, K, V and do it 4 times we will get 4 matrices more or less with equally distributed rows. What is the point of such assignment?
Yael Hamrani — 7:59 AM
To understand that fact exactly, that the embedding doesn’t change. That was the question, do the experiment report what you get and explain why.

Have you got meaningful results for flan-t5 LoRA? I am getting complete gibberish (even after setting bf16=True)
Aviad Stoler — Yesterday at 12:27 PM
This does solve the training issue, so i get a "normal" training curve
the other problem is what this model was design to do https://huggingface.co/docs/transformers/model_doc/flan-t5
the prompt structure is a second issue 
'''
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

inputs = tokenizer("A step by step recipe to make bolognese pasta:", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['Pour a cup of bolognese into a large bowl and add the pasta']
'''
what i got from it 
'''
Question: What is a secured loan?
Level: student

Base output:
“A secured loan is a loan for a student to apply for a loan.”

Fine-tuned Flan-T5 output:
“A secured loan is a type of loan that can be used to buy a home or buy a house. It is a type of loan that can be used to buy a house or buy a house.”

Expected answer:
“A secured loan is a type of loan that requires the borrower to pledge an asset, such as a car or a house, as collateral...”
""" 

Is its a good answer nope but its something
FLAN-T5 · Hugging Face
We’re on a journey to advance and democratize artificial intelligence through open source and open science.
FLAN-T5 · Hugging Face
Boris Kogan — Yesterday at 12:35 PM
Can you share your training?
Aviad Stoler — Yesterday at 1:10 PM
what train csv q&a?
Boris Kogan — Yesterday at 1:12 PM
Yes. Training data and code. Just want to compare. 10x
Mor Sondak
OP
 — Yesterday at 1:44 PM
Sorry for jumping in, I can share that what worked for me was  reducing the evaluation set size and increasing the number of epochs. After making these changes the answers weren’t amazing but at least they weren’t completely gibberish.
Boris Kogan — Yesterday at 1:58 PM
Thank you. How many examples and epochs?
Mor Sondak
OP
 — Yesterday at 2:01 PM
My training set has 1700 examples, I reduced the evaluation set to 0.05. It’s not the best practice, but I felt I am losing valuable data. And 6 epochs.
