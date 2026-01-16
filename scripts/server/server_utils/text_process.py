# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import inflect
from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer

# use_ttsfrd = False
from text_preprocess_utils import (
    contains_chinese,
    replace_blank,
    replace_corner_mark,
    remove_bracket,
    spell_out_number,
    is_digits_only
)


zh_tn_model = ZhNormalizer(remove_erhua=False, full_to_half=False)
inflect_parser = inflect.engine()
en_tn_model = EnNormalizer()

import re
from nltk.corpus import words
import nltk 
nltk.data.path.append('./nltk_data')


# 从NLTK中获取常见单词列表
word_list = set(words.words())


def check_reading_mode(text: str):
    # 匹配所有连续的英文字母序列
    english_sequences = re.findall(r"[a-zA-Z]+", text)

    # result = {}
    for seq in english_sequences:
        # 判断这个序列是否是一个常见单词
        if (seq.lower() not in word_list) or (seq.isupper() and len(seq) <= 2):
            text = text.replace(seq, " ".join(seq))
        else:
            text = text.replace(seq, seq.lower())
            # result[seq] = "可能需要逐字读出"
    return text


def text_normalize(
    text: str,
    split=True,
    token_max_n=80,
    token_min_n=60,
    merge_len=20,
    comma_split=False,
    allowed_special="all",
):
    text = text.strip().lower()
    if  contains_chinese(text) or is_digits_only(text):
        text = text.replace("：", "。")
        text = text.replace("**", "")
        text = text.replace("（", "，")
        text = text.replace("）", "，")
        text = zh_tn_model.normalize(text)
        text = text.replace("\n", "")
        text = replace_blank(text)
        text = replace_corner_mark(text)

        text = text.replace('"', "")
        text = text.replace("“", "")
        text = text.replace("”", "")
        text = text.replace(" - ", "，")
        text = remove_bracket(text)
        text = re.sub(r"[，,]+$", "。", text)
        # text = check_reading_mode(text)
       
    else:
        # pass
        # text = en_tn_model.normalize(text)
        # # text = check_reading_mode(text)
        text = spell_out_number(text, inflect_parser)        
    return text
if __name__ == '__main__':
    import time 
    start = time.time()
    print(text_normalize(text="hello,my number is 135,678,922,34"))
    end = time.time()
    print(f"time is {(end-start)*1000:.2f}毫秒")