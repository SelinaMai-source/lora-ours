from core.formatting import format_citb_t5
instruction = "Translate to French."
input_text = "Hello world"
print(">>>", format_citb_t5(instruction, input_text), "<<<")
