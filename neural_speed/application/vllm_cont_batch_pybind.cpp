//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
// Defines sigaction on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <stdlib.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <random>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.h"
#include "core/layers/bestla_common.hpp"
#include "core/layers/bestla_gemm.h"
#include "bestla/bestla_parallel.h"
#include "models/model_utils/model_types.h"
#include "models/model_utils/model_config.h"
#include "models/model_utils/model_utils.h"
#include "models/model_utils/quant_utils.h"
#include "models/model_utils/scheduler.h"
#include "models/model_utils/pool.h"

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <signal.h>
#include <windows.h>
#endif

namespace py = pybind11;

// namespace {
// struct Query {
//   uint64_t id;
//   std::vector<model_vocab::id> token_ids;
//   uint64_t max_new_tokens;
//   Query() {}
//   Query(uint64_t id, const pybind11::array_t<model_vocab::id, py::array::c_style | py::array::forcecast>& token_ids, uint64_t max_new_tokens)
//       : id(id), token_ids(token_ids.data(), token_ids.data() + token_ids.size()), max_new_tokens(max_new_tokens) {
//     assert(token_ids.ndim() == 1 || (token_ids.ndim() == 2 && token_ids.shape(0) == 1));
//     assert(max_new_tokens > 0);
//   }

//   std::string to_string() const {
//     const std::string repr_ids(py::str(py::array_t<int, py::array::c_style>(token_ids.size(), token_ids.data())));
//     return std::to_string(id) + ": " + repr_ids + ": " + std::to_string(max_new_tokens);
//   }
// };
// // Response happens to be the same structure as Query, while the tokens IDs is for prompt in a Query but is for
// // generated tokens in a Response.
// using Response = Query;
// using ResponseCallback = std::function<void(std::vector<Response>, int)>;
// }  // namespace

enum GenerationStatus {
  GENERATION_STATUS_UNKNOWN = 0,
  GENERATION_STATUS_WAITING = 1, // set after being converted to sequence
  GENERATION_STATUS_RUNNING = 2,
  GENERATION_STATUS_FINISHED = 3,
};

struct Generation {
  int query_id;
  int n_prompt_tokens;
  int n_generated_tokens;
  int max_new_tokens;
  int64_t receive_time;
  int64_t end_time;
  GenerationStatus status;

  int32_t * prompt_ids;
  int32_t * generated_ids;

  Generation(int max_prompt_tokens, int global_max_new_tokens) {
    reset(GENERATION_STATUS_UNKNOWN);
    prompt_ids = new int32_t[max_prompt_tokens];
    generated_ids = new int32_t[global_max_new_tokens];
  }

  reset(GenerationStatus s) {
    query_id = -1;
    n_prompt_tokens = 0;
    n_generated_tokens = 0;
    max_new_tokens = 0;
    receive_time = 0;
    end_time = 0;
    status = s;
  }

  ~Generation() {
    if (prompt_ids) delete[] prompt_ids;
    if (generated_ids) delete[] generated_ids;
  }

};

class GenerationPool {
 public:
  GenerationPool(int pool_size, int n_prompt_ids, int n_new_tokens) : pool_size(pool_size), n_prompt_ids(n_prompt_ids), n_new_tokens(n_new_tokens) {
    generations = new Generation[pool_size];
    for (int i = 0; i < pool_size; i++) {
      generations[i] = new Generation(n_prompt_ids, n_new_tokens);
    }
  }

  ~GenerationPool() {
    for (int i = 0; i < pool_size; i++) {
      delete generations[i];
      generations[i] = nullptr;
    }
    delete generations;
    generations = nullptr;
    query_id_2_gen_id.clear();
    generations.clear();
  }

  Generation* get_generations() {
    return generations;
  }

  // return address for python ctypes
  unsigned long get_generations_address() {
    return (unsigned long)generations;
  }

  // return address for python ctypes
  // get free slot for new queries. generation ids are cumulated in new_generation_ids
  unsigned long get_free_generation_slot_address(int query_id) {
    int index = next_free_gen_id;
    for (; index < pool_size; index++) {
      if (generations[index].query_id == -1 || generations[index].status < GENERATION_STATUS_RUNNING) { // should be no unknown and waiting generation since these status are managed in vllm
        return (unsigned long)occupy_slot(query_id, index);
      }
    }
    index = 0;
    for (; index < next_free_gen_id; index++) {
      if (generations[index].query_id == -1) {
        return (unsigned long)occupy_slot(query_id, index);
      }
    }
    return nullptr;
  }

  void clear_generation_by_gen_id(int gen_id, GenerationStatus status) {
    if (gen_id < 0 || gen_id >= pool_size) {
      return;
    }
    generations[gen_id].reset(status);
  }

  void clear_generation_by_query_id(int query_id, GenerationStatus status) {
    if (query_id_2_gen_id.find(query_id) == query_id_2_gen_id.end()) {
      fprintf("ERROR: query_id %d not found in query_id_2_gen_id\n", query_id);
      return;
    }
    clear_generation_by_gen_id(query_id_2_gen_id[query_id], status);
    query_id_2_gen_id.erase(query_id);
  }

  void mark_generation_done_by_query_id(int query_id) {
    clear_generation_by_query_id(query_id, GENERATION_STATUS_FINISHED);
  }

  int get_pool_size() {
    return pool_size;
  }

  std::vector<int>& get_new_generation_ids() {
    return new_generation_ids;
  }

  std::ordered_map<int, int>& get_query_id_2_gen_id() {
    return query_id_2_gen_id;
  }

  // completed generations

 private:

  Generation* occupy_slot(int query_id, int index) {
    generations[index].query_id = query_id;
    query_id_2_gen_id[query_id] = index;
    new_generation_ids.push_back(index);
    next_free_gen_id = (index == (pool_size - 1) ? 0 : index + 1);
    assert(("Generation not consumed", generations[index].status != GENERATION_STATUS_CONSUMED));
    generations[index].status = GENERATION_STATUS_UNKNOWN;
    return &generations[index];
  }

  int pool_size;
  int n_prompt_ids;
  int n_new_tokens;
  int next_free_gen_id;
  Generation * generations;
  std::vector<int> new_generation_ids; // cleared after each batch_beam_generate
  std::ordered_map<int, int> query_id_2_gen_id;
};

static std::set<model_archs> cont_batching_model_archs = {MODEL_GPTJ, MODEL_LLAMA};
void init_gpt_params(gpt_params* params, const std::string& model_path, int max_new_tokens = -1, int n_batch = 512,
                     int ctx_size = 512, int seed = -1, int threads = 8, float repetition_penalty = 1.1f,
                     int num_beams = 1, bool do_sample = false, int top_k = 40, float top_p = 0.95,
                     float temperature = 0.8, int min_new_tokens = 0, float length_penalty = 1.0f,
                     bool early_stopping = false, int n_keep = 0, int n_discard = -1, bool shift_roped_k = false,
                     int batch_size = 1, model_vocab::id pad_token = -1, const std::string& memory_dtype = "auto",
                     bool continuous_batching = true, const int& max_request_num = MODEL_MAX_REQUEST_NUM,
                     const float& scratch_size_ratio = 1.0f) {
  MODEL_ASSERT(params != nullptr);
#ifdef MODEL_NAME
  params->model_name = MODEL_NAME;
#endif
  params->model_arch = model_name_to_arch::init().find(params->model_name);
  params->model = model_path;
  params->n_predict = max_new_tokens;
  params->n_batch = n_batch;
  params->n_ctx = ctx_size;
  params->seed = seed;
  params->n_threads = threads;
  params->repeat_penalty = repetition_penalty;
  params->beam_size = num_beams;
  params->do_sample = do_sample;
  params->batch_size = batch_size;
  params->beam_search = (num_beams > 1 && !do_sample);
  params->top_k = top_k;
  params->top_p = top_p;
  params->temp = temperature;
  params->n_keep = n_keep;
  params->n_discard = n_discard;
  params->shift_roped_k = shift_roped_k;
  if (memory_dtype == "f32")
    params->memory_type = KV_MEM_TYPE_F32;
  else if (memory_dtype == "f16")
    params->memory_type = KV_MEM_TYPE_F16;
  else if (memory_dtype == "auto")
    params->memory_type = KV_MEM_TYPE_AUTO;
  else
    fprintf(stderr, "Unexpected memory dtype %s!", memory_dtype.c_str());

  // TODO(Yi & YZT): MHA IN MULTI-BATCH For More Model Archs
  params->cont_batching = continuous_batching;
  if (params->shift_roped_k) params->cont_batching = false;
  if (cont_batching_model_archs.count(params->model_arch) == 0) params->cont_batching = false;
  if (batch_size > 1 && !continuous_batching) {
    params->memory_type = KV_MEM_TYPE_F16;
  }
  params->max_request_num = std::max(batch_size, max_request_num);
  params->min_new_tokens = min_new_tokens;
  params->length_penalty = length_penalty;
  params->do_early_stopping = early_stopping;
  params->scratch_size_ratio = scratch_size_ratio;

  // TODO(Yi): MHA FOR LONG TOKENS
  int32_t tokens_length = 6144;
  if (params->n_ctx > tokens_length) {
    params->memory_type = KV_MEM_TYPE_F16;
  }

  printf(
      "beam_size: %d, do_sample: %d, top_k: %d, top_p: %.3f, continuous_batching: %d, max_request_num: %d, "
      "early_stopping: %d, scratch_size_ratio: %.3f\n",
      params->beam_size, params->do_sample, params->top_k, params->top_p, params->cont_batching,
      params->max_request_num, params->do_early_stopping, params->scratch_size_ratio);
}

std::shared_ptr<quant_layer_base> get_model_quant_layer(const std::string model_name) {
  return ql_registry::create_ql(model_name);
}

#define STATIC_INPUT_HEAD_IDX 0
class Model {

 public:
  Model() { model_init_backend(); }

  ~Model() {
    if (ctx) model_free(ctx);
    if (bsf) delete bsf;
    if (generation_pool) delete generation_pool;
  }

  void init_model(const std::string& model_path, ResponseCallback& response_callback, int max_new_tokens, int n_batch, int ctx_size, int seed, int threads,
                  float repetition_penalty, int num_beams, bool do_sample, int top_k, float top_p, float temperature,
                  int min_new_tokens, float length_penalty, bool early_stopping, int n_keep, int n_discard,
                  bool shift_roped_k, int batch_size, int max_batched_tokens, model_vocab::id pad_token, const std::string& memory_dtype,
                  bool continuous_batching, const int& max_request_num, const float& scratch_size_ratio);

  std::vector<int> batch_beam_generate();

  unsigned long get_generations_address() { return generation_pool->get_generations_address(); }

  unsigned long get_free_generation_slot_address(int query_id) { return generation_pool->get_free_generation_slot_address(query_id); }

  model_token get_eos_id() { return ctx->vocab.eos_token_id; }

  static int quant_model(const std::string& model_path, const std::string& out_path, const std::string& weight_dtype,
                         const std::string& alg, int group_size, const std::string& scale_dtype,
                         const std::string& compute_dtype, bool use_ggml, int threads);

  void print_time() { model_print_timings(ctx); }

  void reset_time() { model_reset_timings(ctx); }

 private:
  model_context* ctx = nullptr;
  gpt_params params;

  // ====NS
  GenerationPool * generation_pool = nullptr;
  // ====NS support continuous batching, beam-search for now.
  beam_search_flow* bsf = nullptr;
  std::vector<bool> free_req_idx;
  std::vector<sequence> running_seqs;
  std::unordered_map<int, int> reqidx_to_vecid;
  std::vector<int> request_done_ids;
  // ResponseCallback response_callback;
  bool prepare_batch_inputs(std::vector<sequence> *seqs, const int &n_input, model_input *inputs);
  bool batch_step(std::vector<sequence> *seqs, const int &n);
  bool batch_done();
  bool update_seqs(std::vector<sequence> *seqs, const int &n_input);
  int query_free_req_idx();

  sequence Generation2Sequence(const Generation& g) {
    sequence ret_seq;
    ret_seq.request_idx = query_free_req_idx();
    ret_seq.prompt_ids.assign(g.prompt_ids, g.prompt_ids + g.n_prompt_tokens);
    ret_seq.n_prompt_tokens = g.n_prompt_tokens;
    ret_seq.n_tokens = g.g.n_prompt_tokens;
    ret_seq.n_past = 0;
    ret_seq.n_total = 0;
    ret_seq.gen_conf.max_new_tokens = g.max_new_tokens;
    ret_seq.gen_conf.min_new_tokens = params.min_new_tokens;
    ret_seq.gen_conf.length_penalty = params.length_penalty;
    ret_seq.gen_conf.do_early_stopping = params.do_early_stopping;
    ret_seq.query_id = g.query_id;
    g.status = GENERATION_STATUS_WAITING;
    return ret_seq;
  }

  void updateGeneration(const sequence& seq, Generation* gs, int gen_id) {
    Generation& g = gs[gen_id];
    assert(("generation id should be the same as query id", g.query_id == seq.query_id));
    int ret_size = seq.generated_ids.size();
    g.n_generated_tokens = ret_size;
    std::copy(seq.generated_ids.cbegin(), seq.generated_ids.cend(), ret_query.token_ids.begin());
    // reset generation
    gs->mark_generation_done_by_query_id(g.query_id);
  }
};

void Model::init_model(const std::string& model_path, int max_new_tokens, int n_batch, int ctx_size, int seed,
                       int threads, float repetition_penalty, int num_beams, bool do_sample, int top_k, float top_p,
                       float temperature, int min_new_tokens, float length_penalty, bool early_stopping, int n_keep,
                       int n_discard, bool shift_roped_k, int batch_size, int max_batched_tokens, model_vocab::id pad_token,
                       const std::string& memory_dtype, bool continuous_batching, const int& max_request_num,
                       const float& scratch_size_ratio) {
  init_gpt_params(&params, model_path, max_new_tokens, n_batch, ctx_size, seed, threads, repetition_penalty, num_beams,
                  do_sample, top_k, top_p, temperature, min_new_tokens, length_penalty, early_stopping, n_keep,
                  n_discard, shift_roped_k, batch_size, pad_token, memory_dtype, continuous_batching, max_request_num,
                  scratch_size_ratio);
  ctx = model_init_from_gpt_params(params);
  if (pad_token != -1) ctx->vocab.pad_token_id = pad_token;

  // ====NS
  if (generation_pool == nullptr) {
    generation_pool = new GenerationPool(max_request_num, ctx->cxt_size, ctx->max_new_tokens);
  }

  // ====NS support continuous batching, beam-search for now.
  // this->response_callback = response_callback;
  if (ctx->beam_search && bsf == nullptr) {
    bsf = new beam_search_flow(ctx, ctx->max_request_num, params.n_threads);
  }
  free_req_idx.resize(ctx->max_request_num, true);
}

int Model::query_free_req_idx() {
  auto iter = std::find_if(free_req_idx.begin(), free_req_idx.end(), [](const bool flag) { return flag; });
  if (iter != free_req_idx.end()) {
    int idx = std::distance(free_req_idx.begin(), iter);
    free_req_idx[idx] = false;
    return idx;
  }
  return -1;
}

bool Model::batch_done() {
  return !request_done_ids.empty();
}

bool Model::prepare_batch_inputs(std::vector<sequence> *seqs, const int &n_input, model_input *inputs) {
  for (int i = 0; i < n_input; ++i) {
    if ((seqs->at(i)).status != seq_status::PREFILL && (seqs->at(i)).status != seq_status::DECODING) {
      fprintf(stderr, "%s: error: request %d status is unright (%d).\n", __func__, seqs->at(i).request_idx,
              static_cast<int>((seqs->at(i)).status));
      return false;
    } else if ((seqs->at(i)).status == seq_status::PREFILL) {
      inputs[i].tokens = (seqs->at(i)).prompt_ids.data();
      inputs[i].n_tokens = (seqs->at(i)).n_prompt_tokens;
      inputs[i].n_prompt_tokens = (seqs->at(i)).n_prompt_tokens;
      inputs[i].n_past = 0;
      inputs[i].n_total = 0;
      inputs[i].request_idx = (seqs->at(i)).request_idx;
      // do not support padding for now
      inputs[i].n_padding = 0;
      inputs[i].gen_conf = (seqs->at(i)).gen_conf;
    } else if ((seqs->at(i)).status == seq_status::DECODING) {
      inputs[i].tokens = &(seqs->at(i)).generated_ids.back();
      inputs[i].n_tokens = 1;
      inputs[i].n_past = (seqs->at(i)).n_past;
      inputs[i].n_total = (seqs->at(i)).n_total;
      inputs[i].request_idx = (seqs->at(i)).request_idx;
      // do not support padding for now
      inputs[i].n_padding = 0;
    } else {
      continue;
    }
  }
  return true;
}

bool Model::update_seqs(std::vector<sequence> *seqs, const int& n_input) {
  request_done_ids.clear();
  for (int ni = 0; ni < n_input; ++ni) {
    if (seqs->at(ni).status == seq_status::PREFILL) {
      seqs->at(ni).status = seq_status::DECODING;
      seqs->at(ni).n_past = seqs->at(ni).n_prompt_tokens;
      seqs->at(ni).n_total = seqs->at(ni).n_prompt_tokens;
      seqs->at(ni).n_tokens = 1;
    } else if (seqs->at(ni).status == seq_status::DECODING) {
      seqs->at(ni).n_tokens = 1;
      seqs->at(ni).n_past += seqs->at(ni).n_tokens;
      seqs->at(ni).n_total += seqs->at(ni).n_tokens;
    } else {
      fprintf(stderr, "%s: error: wrong sequence status %d.\n", __func__, static_cast<int>(seqs->at(ni).status));
      return false;
    }
  }
  if (ctx->beam_search && bsf != nullptr) {
    request_done_ids = bsf->request_done_ids();
    std::vector<std::vector<model_token>> req_done_res = bsf->request_done_reponse();
    if (request_done_ids.size() != req_done_res.size()) {
      fprintf(stderr,
              "%s: error: beam search give mis-matched size between finished request ids and generated "
              "tokens.\n",
              __func__);
      return false;
    }
    for (int r = 0; r < request_done_ids.size(); ++r) {
      const int idx = request_done_ids[r];
      if (reqidx_to_vecid.count(idx) == 0) {
        fprintf(stderr, "%s: error: done request idx: %d not in executed_seqs.\n", __func__, idx);
        return false;
      }
      const int vecid = reqidx_to_vecid[idx];
      seqs->at(vecid).generated_ids = std::move(req_done_res[r]);
      seqs->at(vecid).status = seq_status::FINISHED;
      seqs->at(vecid).end_time = model_time_us();
    }
    return true;
  }
  return false;  // TODO (YZT) greedy search and top_p-top_k sampling
}

bool Model::batch_step(std::vector<sequence> *seqs, const int &n) {
  reqidx_to_vecid.clear();
  for (int ni = 0; ni < seqs->size(); ++ni) {
    reqidx_to_vecid.emplace(seqs->at(ni).request_idx, ni);
  }

  std::vector<model_input> step_inputs(n);
  if (!prepare_batch_inputs(seqs, n, step_inputs.data())) {
    return false;
  }
  if (!bsf->step(step_inputs)) {
    return false;
  }
  return update_seqs(seqs, n);
}

std::vector<int> Model::batch_beam_generate() {
  if (ctx->beam_search) {
    std::vector<sequence> seqs;
    std::vector<int>& new_generation_ids = generation_pool->get_new_generation_ids();
    int n_new_queries = new_generation_ids.size(); // used later
    seqs.reserve(n_new_queries + running_seqs.size());
    // add existing seqs
    for (const auto& seq : running_seqs) {
      if (reqidx_to_vecid.find(seq.request_idx) == reqidx_to_vecid.end()) {
        fprintf(stderr, "%s: error: request idx: %d not found in reqidx_to_vecid.\n", __func__, seq.request_idx);
        exit(0);
      }
      seqs.push_back(seq);
    }
    // add new queries as sequence
    Generation *gs = generation_pool->get_generations();
    int n_new_queries;
    for (const auto& id : new_generation_ids) {
      seqs.push_back(Generation2Sequence(gs[id])); // generation status is set to WAITING
      sequence& seq = seqs.back();
      if (reqidx_to_vecid.find(seq.request_idx) != reqidx_to_vecid.end()) {
        fprintf(stderr, "%s: error: request idx: %d already used.\n", __func__, seq.request_idx);
        exit(0);
      }
    }
    // clear new generation ids for next round of generation
    new_generation_ids.clear();

    assert(("number of request should not be 0 and not exceed max_request_num", seqs.size() <= ctx->max_request_num && seqs.size > 0));
    
    if (seqs.size() < ctx->max_request_num) { // execute one step
      batch_step(&seqs, seqs.size());
    } else { // execute multiple steps due to no slot
      while (!batch_done()) {
        batch_step(&seqs, seqs.size());
      }
    }
    // update Generations
    if (!request_done_ids.empty()) {
      std::unordered_map<int, int>& query_id_2_gen_id = generation_pool->get_query_id_2_gen_id();
      std::vector<int> done_gen_ids;
      done_gen_ids.reserve(request_done_ids.size());
      for (int r = 0; r < request_done_ids.size(); ++r) {
        const int idx = request_done_ids[r];
        if (reqidx_to_vecid.find(idx) == reqidx_to_vecid.end()) {
          fprintf(stderr, "%s: error: done request idx: %d not in executed_seqs.\n", __func__, idx);
          exit(0);
        }
        const int vecid = reqidx_to_vecid[idx];
        const int genid = query_id_2_gen_id[seqs[vecid].query_id];
        done_gen_ids.push_back(genid);
        // done_queries.emplace_back(idx, seqs[vecid].generated_ids, seqs[vecid].gen_conf.max_new_tokens);
        Sequence2Generation(seqs[vecid], gs, genid); // copy generated tokens and reset generation
        reqidx_to_vecid.erase(idx);
        free_req_idx[idx] = true; // release request idx and corresponding kv cache will be overwritten
        running_seqs.erase(running_seqs.begin() + vecid - n_new_queries);
      }
      // no callback for now
      // if (response_callback) {
      //   std::vector<Query> done_queries;
      //   done_queries.reserve(request_done_ids.size());
      //   std::transform(done_queries_map.begin(), done_queries_map.end(), std::back_inserter(done_queries),
      //                  [](auto &pair) { return pair.second; });
      //   response_callback(done_queries, done_queries.size());
      // }
      return done_gen_ids;
    }
    // no done request
    return {};
  }
  fprintf(stderr, "\nERROR: Only beam search supported for now!\n");
  exit(0);
}

int Model::quant_model(const std::string& model_path, const std::string& out_path, const std::string& weight_dtype,
                       const std::string& alg, int group_size, const std::string& scale_dtype,
                       const std::string& compute_dtype, bool use_ggml, int threads) {
  quant_params q_params;
#ifdef MODEL_NAME
  q_params.model_name = MODEL_NAME;
#endif
  model_archs mt = model_name_to_arch::init().find(q_params.model_name);
  if (mt == MODEL_UNKNOWN) {
    fprintf(stderr, "error, please set model_name \n");
    exit(0);
  }
  q_params.model_arch = mt;
  q_params.model_file = model_path;
  q_params.out_file = out_path;
  q_params.weight_dtype = weight_dtype;
  q_params.alg = alg;
  q_params.group_size = group_size;
  q_params.scale_dtype = scale_dtype;
  q_params.compute_dtype = compute_dtype;
  q_params.use_ggml = use_ggml;
  q_params.nthread = threads;

  ne_ftype ftype = quant_params_to_ftype(q_params);
  printf("ne_ftype: %d\n", ftype);

  auto quant_layer = get_model_quant_layer(q_params.model_name);
  if (model_quantize(q_params, quant_layer)) {
    fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, q_params.model_file.c_str());
    return 1;
  }
  return 0;
}

// empty response funtion as default callback value
// void empty_response(std::vector<Response>, int) {}
// ResponseCallback empty_response_callback = empty_response;

#if MODEL_NAME_ID == 1

PYBIND11_MODULE(gptj_vllm_cb_cpp, m)

#elif MODEL_NAME_ID == 5

PYBIND11_MODULE(llama_vllm_cb_cpp, m)

#endif
{
  m.doc() = "cpp model python binding";
  py::class_<Model>(m, "Model", py::module_local())
      .def(py::init())
      .def("init_model", &Model::init_model, "initial model with model path and parameters", py::arg("model_path"),
           py::arg("max_new_tokens") = -1, py::arg("n_batch") = 512, py::arg("ctx_size") = 1024, py::arg("seed") = -1,
           py::arg("threads") = 8, py::arg("repetition_penalty") = 1.1f, py::arg("num_beams") = 1,
           py::arg("do_sample") = false, py::arg("top_k") = 40, py::arg("top_p") = 0.95, py::arg("temperature") = 0.8,
           py::arg("min_new_tokens") = 0, py::arg("length_penalty") = 1.0, py::arg("early_stopping") = false,
           py::arg("n_keep") = 0, py::arg("n_discard") = -1, py::arg("shift_roped_k") = false,
           py::arg("batch_size") = 1, py::arg("max_batched_tokens") = -1, py::arg("pad_token") = -1,
           py::arg("memory_dtype") = "auto",
           py::arg("continuous_batching") = true, py::arg("max_request_num") = MODEL_MAX_REQUEST_NUM,
           py::arg("scratch_size_ratio") = 1.0f)
      .def("batch_beam_generate", &Model::batch_beam_generate, "batch beam generate")
      .def_static("quant_model", &Model::quant_model, "Quantize model", py::arg("model_path"), py::arg("out_path"),
                  py::arg("weight_dtype") = "int4", py::arg("alg") = "sym", py::arg("group_size") = 32,
                  py::arg("scale_dtype") = "fp32", py::arg("compute_dtype") = "int8", py::arg("use_ggml") = false,
                  py::arg("threads") = 8)
      .def("print_time", &Model::print_time)
      .def("reset_time", &Model::reset_time)
      .def("get_eos_id", &Model::get_eos_id)
      .def("get_generations_address", &Model::get_generations_address)
      .def("get_free_generation_slot_address", &Model::get_free_generation_slot_address)
      ;
  // py::class_<Query>(m, "Query")
  //     .def(py::init<uint64_t, py::array_t<model_vocab::id>, uint16_t>())
  //     .def("__repr__", &Query::to_string)
  //     .def_readwrite("id", &Query::id)
  //     .def_readwrite("token_ids", &Query::token_ids)
  //     .def_readwrite("max_new_tokens", &Query::max_new_tokens);


  
}


