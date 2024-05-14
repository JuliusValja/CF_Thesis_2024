import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    logging,
)
from accelerate import Accelerator
from datasets import load_dataset
import pandas as pd
import numpy as np
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import re
from sklearn import metrics
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import os
import time


sys = """You are evaluating counterfactual explanations generated by AI.
        Counterfactual explanations explain what parameters of a situation should have been different for the outcome to have been different.
        You are not expected to provide reasoning or explanation and should answer with the appropriate value from the set ["low", "medium", "high"]."""

sysend =  "The following is the counterfactual explanation. "

#Choosing the correct instruction for metric
def match1(i):
    if i==0: 
        return " Please rate as 'low' (very unsatisfied), 'medium' or 'high' (completely satisfied), how satisfied would you be with such an explanation: "
    if i==1: 
        return " Please rate as 'low' (very unfeasible), 'medium' or 'high' (completely feasible), how feasible is this explanation: "
    if i==2: 
        return " Please rate as 'low' (very unconsistent), 'medium' or 'high' (completely consistent), how consistent is this explanation: "
    if i==3:
        return " Please rate as 'low' (very incomplete), 'medium' or 'high' (fully complete), how complete is this explanation: "
    if i==4:
        return " Please rate as 'low' (not at all), 'medium' or 'high' (completely), how much would you trust this explanation: "
    if i==5:
        return " Please rate as 'low' (incomprehensible), 'medium' or 'high' (very understandable), how understandable is this explanation: "
    if i==6:
        return " Please rate as 'low' (completely biased), 'medium' or 'high' (completely fair), how fair is this explanation: "
    if i==7:
        return " Please rate as 'low' (too simple), 'medium' (ideally complex) or 'high' (too complex), how complex is this explanation: "
    return " Please rate as 'low' (very unsatisfied), 'medium' or 'high' (completely satisfied), how satisfied would you be with such an explanation: "

#Choosing the correct definition for metric
def match2(i):
    if i==0:
        return " The definition of satisfaction: this scenario effectively explains how to reach a different outcome. "
    if i==1:
        return " The definition of feasibility: the actions suggested by the explanation are practical, realistic to implement and actionable. "
    if i==2:
        return " The definition of consistency: the parts of the explanation do not contradict each other. "
    if i==3:
        return " The definition of completeness: the explanation is sufficient in explaining how to achieve the desired outcome. "
    if i==4:
        return " The definition of trust: I believe that the suggested changes would bring about the desired outcome. "
    if i==5:
        return " The definition of understandability:  I feel like I understood the phrasing of the explanation well. "
    if i==6:
        return " The definition of fairness: the explanation is unbiased towards different user groups and does not operate on sensitive features. "
    if i==7:
        return " The definition of complexity: the explanation has an appropriate level of detail and complexity - not too simple, yet not overly complex."
    return " The definition of satisfaction: this scenario effectively explains how to reach a different outcome. "
        
#Function for creating the base prompt
def prompt_base(s,i,answer, llama):
    global sys
    global sysend
    prompt = s + match1(i)
    if(answer!=None):
        if(llama):
            final = [
                    {"role":"system", "content":sys + match2(i) + sysend},
                    {"role":"user", "content":prompt},
                    {"role":"assistant", "content": answer}
                ]
        else:
            final = [
                    {"role":"user", "content":sys + match2(i) + sysend + prompt},
                    {"role":"assistant", "content": answer}
                ]
    else:
        if(llama):
            final = [
                    {"role":"system", "content":sys + match2(i) + sysend},
                    {"role":"user", "content":prompt}
                ]
        else:
            final = [
                    {"role":"user", "content":sys + match2(i) + sysend + prompt}
                ]
    return final


#Function for creating the prompt with all definitions
def prompt_defs(s,i,answer, llama):
    global sys
    global sysend
    defs =  """The definition of satisfaction: this scenario effectively explains how to reach a different outcome. The definition of feasibility: the actions suggested by the explanation are practical, realistic to implement and actionable. The definition of consistency: the parts of the explanation do not contradict each other. The definition of completeness: the explanation is sufficient in explaining how to achieve the desired outcome. The definition of trust: I believe that the suggested changes would bring about the desired outcome. The definition of understandability:  I feel like I understood the phrasing of the explanation well. The definition of fairness: the explanation is unbiased towards different user groups and does not operate on sensitive features. The definition of complexity: the explanation has an appropriate level of detail and complexity - not too simple, yet not overly complex. """
    prompt = s + match1(i)
    if(answer!=None):
        #train_answer = " I would rate the " + d[i] + " of this explanation as " + answer
        if(llama):
            final = [
                    {"role":"system", "content":sys + defs + sysend},
                    {"role":"user", "content":prompt},
                    {"role":"assistant", "content": answer}
                ]
        else:
            final = [
                    {"role":"user", "content":sys + defs + sysend + prompt},
                    {"role":"assistant", "content": answer}
                ]
    else:
        if(llama):
            final = [
                    {"role":"system", "content":sys + defs + sysend},
                    {"role":"user", "content":prompt}
                ]
        else:
            final = [
                    {"role":"user", "content":sys + defs + sysend + prompt}
                ]
    return final


#Function for creating the prompt with all definitions and two examples
def prompt_examples(s,i,answer, llama):
    global sys
    global sysend
    defs =  """ The definition of satisfaction: this scenario effectively explains how to reach a different outcome. 
   The definition of feasibility: the actions suggested by the explanation are practical, realistic to implement and actionable. 
   The definition of consistency: the parts of the explanation do not contradict each other. 
   The definition of completeness: the explanation is sufficient in explaining how to achieve the desired outcome. 
   The definition of trust: I believe that the suggested changes would bring about the desired outcome. 
   The definition of understandability:  I feel like I understood the phrasing of the explanation well. 
   The definition of fairness: the explanation is unbiased towards different user groups and does not operate on sensitive features. 
   The definition of complexity: the explanation has an appropriate level of detail and complexity - not too simple, yet not overly complex."""
    examples = """Here are two examples of a prompt and the output. 
   Example prompt 1: "Imagine you are in this scenario: 'You are a 21-year-old person who has an average grade of B. You work part-time for 20 hours per week.'
    Current outcome: Your university application was rejected.    
    'To have your application approved, you would need to make the following changes: Improve your average grade from B to A.' The rest of the values will remain constant. 
    Please rate as 'low' , 'medium' or 'high', how consistent is this explanation: "
   
   Example output 1: "high".
   Example prompt 2: "Imagine you are in this scenario: 'You are a 21-year-old person who has an average grade of B. You work part-time for 20 hours per week.'
    Current outcome: Your university application was rejected.    
    'To have your application approved, you would need to make the following changes: Increase your hours worked per week from 20 to 80.' The rest of the values will remain constant. 
    Please rate as 'low' , 'medium' or 'high', how feasible is this explanation: "
   
   Example output 2: "low".
   Please answer questions in a similar format."""
    prompt = s + match1(i)
    if(answer!=None):
        #train_answer = " I would rate the " + d[i] + " of this explanation as " + answer
        if(llama):
            final = [
                    {"role":"system", "content":sys + defs + examples + sysend},
                    {"role":"user", "content":prompt},
                    {"role":"assistant", "content": answer}
                ]
        else:
            final = [
                    {"role":"user", "content":sys + defs + examples + sysend + prompt},
                    {"role":"assistant", "content": answer}
                ]
    else:
        if(llama):
            final = [
                    {"role":"system", "content":sys + defs + examples + sysend},
                    {"role":"user", "content":prompt}
                ]
        else:
            final = [
                    {"role":"user", "content":sys + defs + examples + sysend + prompt}
                ]
    return final

access_token = "" #personal api key for Huggingface

#Choosing a model to use
base_model = "meta-llama/Llama-2-7b-chat-hf"
base_model = "meta-llama/Meta-Llama-3-70B-Instruct"
base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
base_model = 'mistralai/Mistral-7B-Instruct-v0.1'

pad_length = 600

tokenizer = AutoTokenizer.from_pretrained(base_model, token=access_token, padding_side="right", add_eos_token=False, add_bos_token=True, max_length=pad_length, use_fast=False)

#For Llama 3
tokenizer.pad_token_id = 128001

#For other models
tokenizer.pad_token = tokenizer.eos_token


#Here the data is processed to using one of the prompt functions defined above
def preprocess_function(example, train, llama):
    if(train):
        p = base_prompt(example["prompt"], example["type"], example["answer"], llama)
        pre_token = tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=False)
        example = tokenizer(pre_token, padding="max_length", max_length=pad_length, return_attention_mask=True)
    else:
        p = base_prompt(example["prompt"], example["type"], None, llama)
        pre_token = tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        example = tokenizer(pre_token, padding=False)
    
    example["prompt"] = pre_token
    return example

data = load_dataset("csv", data_files={"train":"train_LM.csv", "test":"test_LM.csv"})

llama=False
if "Llama" in base_model:
    llama=True

data["train"] = data["train"].map(preprocess_function, fn_kwargs={"train":True, "llama":llama} , batched=False)
data["test"] = data["test"].map(preprocess_function, fn_kwargs={"train":False, "llama":llama} , batched=False)


#May not be necessary to define explicitly
accelerator = Accelerator()

#Quantization configuration
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map="auto",
    token=access_token
)

#LoRA configuration
peft_params = LoraConfig(
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    r=32,
    bias="none",
    task_type="CAUSAL_LM",
)



model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_params)

model.config.use_cache = False
model.config.pretraining_tp = 1

training_args = transformers.TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    learning_rate=5e-5,
#    optim="paged_adamw_32bit",
    logging_steps=15,
    eval_steps=25,
    save_total_limit=4,
    output_dir="./results",
    save_strategy='epoch',
    evaluation_strategy='steps',
    report_to="none"
)

#For Llama 3
response_template = "<|start_header_id|>assistant<|end_header_id|>"

#For Llama 2 and Mistral
response_template = "[/INST]"

data_collator=DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, response_template = response_template)


trainer = SFTTrainer(
    model=model,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    args=training_args,
    dataset_text_field="prompt",
    data_collator=data_collator
)

trainer.train()

accuracy_scores = []

print("Starting testing")
model.config.use_cache = True
current = time.time()
outps = []
for j in range(4):
    results = []
    for i,pr in enumerate(data["test"]["prompt"]):
        inputs = tokenizer(pr, return_tensors="pt").to("cuda")
        #streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
        c = model.generate(**inputs, temperature=1, do_sample=True, max_new_tokens=60)
        
        results.append([tokenizer.batch_decode(c, skip_special_tokens=False),data["test"]["answer"][i], data["test"]["type"][i]])

    for i,(x,y,z) in enumerate(results):
        
        #For Llama 2 and Mistral
        an = re.search("high|medium|low",x[0].split("[/INST]")[1].lower())
        
        #For Llama 3
        an = re.search("high|medium|low",x[0].split("assistant")[1].lower())
        
        
        result = "Failed"
        if(an!=None):
            result=an.group()

        results[i].append(result)

    answe = np.array(results, dtype=object)
    
    confusion_matrix = metrics.confusion_matrix(answe[:,1], answe[:,3], labels=["low", "medium", "high"])
    acc = (confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[2][2])/len(answe)
    print(confusion_matrix)
    accuracy_scores.append(acc)
    answe2 = []
    
    for u in results:
        if(u[2]==0):
            answe2.append(u)
    
    answe2 = np.array(answe2, dtype=object)
    confusion_matrix2 = metrics.confusion_matrix(answe2[:,1], answe2[:,3], labels=["low", "medium", "high"])
    acc2 = (confusion_matrix2[0][0] + confusion_matrix2[1][1] + confusion_matrix2[2][2])/len(answe2)
    print("Satisfaction " + str(acc2))
    
    outps = outps + results

print(accuracy_scores)
print((time.time()-current)/192)
pd.DataFrame(np.array(outps, dtype=object)).to_csv("results_ll3_70b_ft.csv")
