# https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/llm-inference-hf.html
# Use intel's openvino. There are different ways to use openvino. I will use the hugging face and optimum method.
# pip install optimum[openvino, nncf]. Note that NNCF is optional. It optimizes the model for small footprint and faster inference

# Start using openvino as a backend for hugging face
from optimum.intel import OVModelForCausalLM    # instead of from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

###################### STEP 1 - Convert the model to openvino IR format #################################################
model_id = "meta-llama/Llama-3.2-3B-Instruct"
OV_model_id = "hf_optimum_auto-Llama-3.2-3B-Instruct"
# export=True means the model is converted to openvino IR format on the fly. load_in_8bit means to optimize through weight compression using NNCF
# Note that for model > 1B parameter, weight compression is applied by default
#model = OVModelForCausalLM.from_pretrained(model_id, export=True, load_in_8bit=True, device_map="auto")
#model.save_pretrained(OV_model_id)

##################### STEP 2
# load model
model = OVModelForCausalLM.from_pretrained(OV_model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

query = "I am a senior manager still under probation. I have just resigned. What is my required notice period?"
context = """All termination of employment must be made in writing. The Company may increase the requisite notice period in the event an employee is promoted to a higher level or critical role, in line with employees in similar roles. In such situations, the increased notice period shall supersede that originally provided in the Letter of Employment.
The table below detailed the required notice period during probation and upon confirmation.
During Probation:
- Senior manager & above : 1 month
- Staff/senior Staff/ Manager: 1 month
Upon Confirmation:
- Senior manager & above : 3 months
- Staff/senior Staff/ Manager: 2 months"""
system_role = f"""You are an expert in HR matters. Please answer the user query with the context given. Make your answer as concise as possible and your answer must come from the context only."
For example:
query: I work the 1st shift. What is my working hours?
answer: Your working hours are from 8am to 5pm
"""

base_prompt = f"""Please answer the following query with the context given.
The context is:
{context}

query: {query}
answer:"""

dialogue_template = [
    {"role": "system", "content": system_role},
    {"role": "user", "content": base_prompt},
]
prompt = tokenizer.apply_chat_template(dialogue_template, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=500)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))
