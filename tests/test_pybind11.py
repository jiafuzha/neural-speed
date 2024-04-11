# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import neural_speed
from neural_speed import ModelForContBatching
import neural_speed.llama_vllm_cb_cpp as cpp
# from neural_speed.llama_vllm_cb_cpp import Sequence
# import numpy as np
from transformers import AutoTokenizer


# a = np.array([1,2,3,4], dtype=np.int64)
# a = np.array([[1,2],[3,4]], dtype=Sequence)
# # a = [Sequence(1, 2), Sequence(2, 4)]
# # # a = [1, 2, 3, 4]
# cpp.update_sequence(a)
# print(a)
# print(len(a))
# print(a.shape)
# tokens = np.array([[1,2,3,4]], np.int32).tolist()

# vs = cpp.VectorSequence()
# vs.append(cpp.Sequence(1, 2))
# vs.append(cpp.Sequence(3, 4))
# print(vs)
# vs[0].a = 100
# print(vs[0].a)

# queries = cpp.QueryVector()
# queries.append(cpp.Query(1, [1,2,3,4], 4))
# queries.append(cpp.Query(2, [1,2,3,4,5], 5))
# cpp.set_queries(queries)
# print(queries)
# prompts = {
#                 "she opened": 4
#         }
# # model_name = "/tf_dataset2/models/nlp_toolkit/llama-2-7b-chat/Llama-2-7b-chat-hf"
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = ModelForContBatching(max_request_num=4, batch_size=4)
# get quantized model
model.init(model_name, use_quant=True, weight_dtype="int4", compute_dtype="int8")
model.load_model()
del model
model_path = "./runtime_outs/ne_llama_q_int4_bestla_cint8_g32.bin"



# tokens = tokenizer("she opened", return_tensors='pt').input_ids.tolist()

# tokens = [1,2,3,4]
# # print(type(tokens))
# id = 0
# v = 4
# q = cpp.Query(id, tokens, v)
# print(q)
# print(tokens)

