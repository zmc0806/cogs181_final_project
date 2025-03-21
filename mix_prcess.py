import os

with open('/Users/zmc/Downloads/cogs181_final_project/data/science_fiction/input.txt', 'r', encoding='utf-8') as f:
    scifi_text = f.read()

with open('/Users/zmc/Downloads/cogs181_final_project/data/romance/input.txt', 'r', encoding='utf-8') as f:
    romance_text = f.read()

# # 保存到各自的目录
# with open('data/scifi/input.txt', 'w', encoding='utf-8') as f:
#     f.write(scifi_text)

# with open('data/romance/input.txt', 'w', encoding='utf-8') as f:
#     f.write(romance_text)

# 创建混合文本
with open('data/mixed/input.txt', 'w', encoding='utf-8') as f:
    # 可以采用交替段落的方式混合，而不是简单连接
    # 这种方式有助于模型学习两种风格之间的过渡
    scifi_paragraphs = scifi_text.split('\n\n')
    romance_paragraphs = romance_text.split('\n\n')
    
    # 取两个文本中段落数较少的那个
    min_paragraphs = min(len(scifi_paragraphs), len(romance_paragraphs))
    
    mixed_text = ""
    for i in range(min_paragraphs):
        mixed_text += scifi_paragraphs[i] + "\n\n"
        mixed_text += romance_paragraphs[i] + "\n\n"
    
    # 如果还有剩余段落，也添加进去
    if len(scifi_paragraphs) > min_paragraphs:
        mixed_text += "\n\n".join(scifi_paragraphs[min_paragraphs:])
    elif len(romance_paragraphs) > min_paragraphs:
        mixed_text += "\n\n".join(romance_paragraphs[min_paragraphs:])
    
    f.write(mixed_text)