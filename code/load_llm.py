from transformers import AutoModelForCausalLM, AutoTokenizer
llm = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
tokenizer.pad_token = tokenizer.eos_token
for param in llm.parameters():
    param.requires_grad = False
print("语言模型加载并冻结成功")