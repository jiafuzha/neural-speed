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

import time
import unittest
import shutil
from neural_speed import Model
import neural_speed.llama_cpp as cpp
from transformers import AutoTokenizer

class TestModelServer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree("./runtime_outs", ignore_errors=True)

    def test_model_server(self):
        # prompts = [
        #         "she opened the door and see",
        #         "tell me 10 things about jazz music",
        #         "What is the meaning of life?",
        #         "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer"\
        #         " The slings and arrows of outrageous fortune, "\
        #         "Or to take arms against a sea of troubles."\
        #         "And by opposing end them. To die—to sleep,",
        #         "Tell me an interesting fact about llamas.",
        #         "What is the best way to cook a steak?",
        #         "Are you familiar with the Special Theory of Relativity and can you explain it to me?",
        #         "Recommend some interesting books to read.",
        #         "What is the best way to learn a new language?",
        #         "How to get a job at Intel?",
        #         "If you could have any superpower, what would it be?",
        #         "I want to learn how to play the piano.",
        #         ]
        # prompts = {
        #         "she opened": 2,
        #         "tell me": 1,
        #         "What is": 3
        # }
        prompts = {
                "she opened": 4
        }
        # model_name = "/tf_dataset2/models/nlp_toolkit/llama-2-7b-chat/Llama-2-7b-chat-hf"
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # model = Model()
        # # get quantized model
        # model.init(model_name, use_quant=True, weight_dtype="int4", compute_dtype="int8")
        # del model
        # model_path = "./runtime_outs/ne_llama_q_int4_bestla_cint8_g32.bin"

        # res_collect = []
        # # response function (deliver generation results and current remain working size in server)
        # def f_response(res, working):
        #     ret_token_ids = [r.token_ids for r in res]
        #     res_collect.extend(ret_token_ids)
        #     ans = tokenizer.batch_decode(ret_token_ids, skip_special_tokens=True,
        #                                 clean_up_tokenization_spaces=False)
        #     print(f"working_size: {working}, ans:", flush=True)
        #     for a in ans:
        #         print(a)
        #         print("=====================================")
        

        # s = cpp.ModelServer(f_response,
        #                         model_path,
        #                         # max_new_tokens=128,
        #                         max_new_tokens=4,
        #                         # num_beams=4,
        #                         num_beams=2,
        #                         # min_new_tokens=30,
        #                         min_new_tokens=4,
        #                         early_stopping=True,
        #                         do_sample=False,
        #                         continuous_batching=True,
        #                         return_prompt=True,
        #                         # max_request_num=8,
        #                         max_request_num=2,
        #                         threads=1,
        #                         print_log=False,
        #                         scratch_size_ratio = 1.0,
        #                         memory_dtype="auto",
        #                         batch_size=1,
        #                     )
        
        tokens = tokenizer("she opened", return_tensors='pt').input_ids.tolist()

        # tokens = [1,2,3,4]
        print(type(tokens))
        q = 0
        v = 4
        tt = cpp.Query(q, tokens, v)
        print(tt)
        # s.issueQuery([cpp.Query(q, tokens, v)])

        #for md in ["auto", "f16"]:
        # for md in ["auto"]:
        #     if md == "auto":
        #         print("=======MHA MODEL SERVER TESTING=========")
        #     else:
        #         print("=======NON-MHA MODEL SERVER TESTING=========")
        #     added_count = 0
        #     s = cpp.ModelServer(f_response,
        #                         model_path,
        #                         # max_new_tokens=128,
        #                         max_new_tokens=4,
        #                         # num_beams=4,
        #                         num_beams=2,
        #                         # min_new_tokens=30,
        #                         min_new_tokens=4,
        #                         early_stopping=True,
        #                         do_sample=False,
        #                         continuous_batching=True,
        #                         return_prompt=True,
        #                         # max_request_num=8,
        #                         max_request_num=2,
        #                         threads=1,
        #                         print_log=False,
        #                         scratch_size_ratio = 1.0,
        #                         memory_dtype= md,
        #                         batch_size=1,
        #                     )
        #     qid = 0
        #     for k, v in prompts.items():
        #         p_token_ids = tokenizer(k, return_tensors='pt').input_ids.tolist()
        #         print(type(p_token_ids))
        #         s.issueQuery([cpp.Query(qid, p_token_ids, v)])
        #         added_count += 1
        #         # time.sleep(2)  # adjust query sending time interval

        #     # recommend to use time.sleep in while loop to exit program
        #     # let cpp server owns more resources
        #     while (added_count != len(prompts) or not s.Empty()):
        #         time.sleep(1)
        #     del s
        #     print("should finished")

if __name__ == "__main__":
    unittest.main()
